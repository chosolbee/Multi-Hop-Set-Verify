import os
import json
import argparse
import torch
import numpy as np
from torch.utils.data import Dataset, random_split, DataLoader, Sampler
from torchmetrics.functional.regression import mean_squared_error, mean_absolute_error, r2_score, spearman_corrcoef, kendall_rank_corrcoef
from torchmetrics.retrieval import RetrievalMRR, RetrievalNormalizedDCG
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EvalPrediction,
    TrainerCallback,
)
from transformers.trainer_utils import PredictionOutput
from config import WANDB_ENTITY, DEBERTA_MAX_LENGTH
import torch.nn as nn
import wandb
import random
from collections import defaultdict

class VerifierDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_length=768, desired_scale=1):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.desired_scale = desired_scale
        self.data = []
        question_ids = []

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                question = entry["question"]
                question_id = entry["question_id"]
                passages = entry["passages"]
                passages_text = " [SEP] ".join([p["paragraph_text"] for p in passages])
                text = f"Question: {question} [SEP] {passages_text}"
                score = entry["score"] * desired_scale
                self.data.append({"text": text, "score": score, "question_id": question_id})
                question_ids.append(question_id)

        unique_question_ids = sorted(set(question_ids))
        question_id_to_idx = {question_id: idx for idx, question_id in enumerate(unique_question_ids)}
        for item in self.data:
            item["question_id"] = question_id_to_idx[item["question_id"]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        encoding["labels"] = {
            "labels": torch.tensor(item["score"], dtype=torch.float),
            "question_id": torch.tensor(item["question_id"], dtype=torch.long),
        }
        return encoding

class GroupSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.groups = defaultdict(list)
        for idx in range(len(dataset)):
            item = dataset[idx]
            q_id = int(item["labels"]["question_id"]) if torch.is_tensor(item["labels"]["question_id"]) else item["labels"]["question_id"]
            self.groups[q_id].append(idx)
        self.group_list = list(self.groups.values())

    def __iter__(self):
        all_batches = []
        for group in self.group_list:
            random.shuffle(group) 
            for i in range(0, len(group), self.batch_size):
                batch = group[i:i + self.batch_size]
                all_batches.append(batch)
        random.shuffle(all_batches)  
        for batch in all_batches:
            yield batch

    def __len__(self):
        total_batches = 0
        for group in self.group_list:
            total_batches += (len(group) + self.batch_size - 1) // self.batch_size
        return total_batches


def collate_fn(batch):
    input_ids = torch.stack([x["input_ids"] for x in batch])
    attention_mask = torch.stack([x["attention_mask"] for x in batch])
    labels = torch.stack([x["labels"]["labels"] for x in batch])
    question_ids = torch.stack([x["labels"]["question_id"] for x in batch])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": {"labels": labels, "question_ids": question_ids},
    }


class Metrics:
    def __init__(self, desired_scale=1):
        self.desired_scale = desired_scale

        self.mrr_metric = RetrievalMRR()
        self.ndcg_metric = RetrievalNormalizedDCG()

    def __call__(self, eval_preds: EvalPrediction):
        logits, labels_dict = eval_preds
        logits = torch.tensor(logits.flatten())
        labels = torch.tensor(labels_dict["labels"].flatten())
        indexes = torch.tensor(labels_dict["question_ids"].flatten())

        predictions = torch.sigmoid(logits)
        labels = labels / self.desired_scale

        spearman_scores = []
        kendall_scores = []
        correct = total = 0

        for idx in torch.unique(indexes):
            mask = indexes == idx
            p = predictions[mask]
            l = labels[mask]

            if len(p) < 2:
                continue

            spearman_scores.append(spearman_corrcoef(p, l).item())
            kendall_scores.append(kendall_rank_corrcoef(p, l).item())

            pred_diff = p.unsqueeze(0) - p.unsqueeze(1)
            label_diff = l.unsqueeze(0) - l.unsqueeze(1)

            pred_sign = torch.sign(pred_diff)
            label_sign = torch.sign(label_diff)

            valid_mask = label_sign != 0
            correct += ((pred_sign == label_sign) & valid_mask).sum().item()
            total += valid_mask.sum().item()

        return {
            "mse": mean_squared_error(predictions, labels).item(),
            "mae": mean_absolute_error(predictions, labels).item(),
            "r2": r2_score(predictions, labels).item(),
            "mrr": self.mrr_metric(predictions, labels, indexes).item(),
            "ndcg": self.ndcg_metric(predictions, labels, indexes).item(),
            "spearman": np.nanmean(spearman_scores),
            "kendall": np.nanmean(kendall_scores),
            "pairwise_accuracy": correct / total if total > 0 else 0,
        }


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.desired_scale = kwargs.pop("desired_scale", 1)
        self.margin = kwargs.pop("margin", 0.1)
        self.alpha = kwargs.pop("alpha", 0.1)
        super().__init__(*args, **kwargs)
        
        self.mse_loss = nn.MSELoss()
        self.ranking_loss_fn = nn.MarginRankingLoss(margin=self.margin)

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: Provided train_dataset is None")
        
        return DataLoader(
            self.train_dataset,
            batch_sampler=GroupSampler(self.train_dataset, self.args.per_device_train_batch_size),
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels_dict = inputs.pop("labels", None)

        outputs = model(**inputs)

        predictions = torch.sigmoid(outputs.logits.squeeze(-1)) * self.desired_scale
        labels = labels_dict["labels"]

        mse = self.mse_loss(predictions, labels)

        # margin ranking loss
        candidate_pairs = []

        for i in range(len(labels)):
            for j in range(len(labels)):
                if labels[i] > labels[j]:
                    candidate_pairs.append((i, j))

        if candidate_pairs:
            pred_i = torch.stack([predictions[i] for i, _ in candidate_pairs])
            pred_j = torch.stack([predictions[j] for _, j in candidate_pairs])
            target = torch.ones_like(pred_i)
            ranking_loss = self.ranking_loss_fn(pred_i, pred_j, target)
        else:
            ranking_loss = torch.tensor(0.0, device=predictions.device)

        loss = mse + self.alpha * ranking_loss

        if return_outputs:
            return (loss, outputs)
        else:
            return loss


class PrintCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            print(f"\n[Step {state.global_step}]")
            for key, value in logs.items():
                print(f"  {key}: {value:.4f}")
            print("-" * 40)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Verifier Train")

    parser.add_argument("--train-data-path", help="Training Dataset Path", type=str, required=True)
    parser.add_argument("--eval-data-path", help="Evaluation Dataset Path", type=str, required=True)
    parser.add_argument("--test-data-path", help="Test Dataset Path", type=str, required=True)
    parser.add_argument("--trainer-output-dir", help="Training Output Path", type=str)
    parser.add_argument("--max-length", help="Max Length of Tokenizer", type=int, default=DEBERTA_MAX_LENGTH)
    parser.add_argument("--desired-scale", help="Desired Scale", type=int, default=1)
    parser.add_argument("--learning-rate", help="Learning Rate", type=float, default=5e-5)
    parser.add_argument("--lr-scheduler-type", help="Learning Rate Scheduler Type", type=str, default="cosine")
    parser.add_argument("--warmup-ratio", help="Warmup Ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", help="Weight Decay", type=float, default=0.01)
    parser.add_argument("--batch-size", help="Batch Size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", help="Gradient Accumulation Steps", type=int, default=4)
    parser.add_argument("--num-epochs", help="Number of Epochs", type=int, default=3)
    parser.add_argument("--fp16", help="Use FP16", action="store_true")
    parser.add_argument("--margin", help="Margin for Ranking Loss", type=float, default=0.1)
    parser.add_argument("--alpha", help="Weight for Ranking Loss", type=float, default=0.1)

    return parser.parse_args()


def main():
    args = parse_arguments()

    model_name = "microsoft/deberta-v3-large"

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank == -1 or local_rank == 0:  # Only initialize wandb on the main process
        wandb.init(
            project="MultiHopVerifierTraining",
            entity=WANDB_ENTITY,
            config={
                "model_name": model_name,
                "max_length": args.max_length,
                "desired_scale": args.desired_scale,
                "learning_rate": args.learning_rate,
                "lr_scheduler_type": args.lr_scheduler_type,
                "warmup_ratio": args.warmup_ratio,
                "weight_decay": args.weight_decay,
                "batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "epochs": args.num_epochs,
                "fp16": args.fp16,
                "margin": args.margin,
                "alpha": args.alpha,
            },
        )
    else:
        os.environ["WANDB_MODE"] = "disabled"  # Disable wandb for other processes

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1, problem_type="regression")

    training_args = TrainingArguments(
        output_dir=args.trainer_output_dir,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=500,
        logging_steps=100,
        report_to=["wandb"],
        load_best_model_at_end=True,
        save_total_limit=3,
        metric_for_best_model="mse",
        greater_is_better=False,
        fp16=args.fp16,
    )

    train_dataset = VerifierDataset(args.train_data_path, tokenizer, args.max_length, args.desired_scale)
    eval_dataset = VerifierDataset(args.eval_data_path, tokenizer, args.max_length, args.desired_scale)

    print_callback = PrintCallback()
    compute_metrics = Metrics(desired_scale=args.desired_scale)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        callbacks=[print_callback],
        desired_scale=args.desired_scale,
        margin=args.margin,
        alpha=args.alpha,
    )

    trainer.train()

    print("Training completed. Evaluating on test dataset...")
    trainer.compute_metrics = Metrics(desired_scale=args.desired_scale)

    test_dataset = VerifierDataset(args.test_data_path, tokenizer, args.max_length, args.desired_scale)
    test_metrics = trainer.evaluate(test_dataset)

    wandb.finish()


if __name__ == "__main__":
    main()
