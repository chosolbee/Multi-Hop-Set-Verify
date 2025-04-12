import os
import json
import argparse
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
from torchmetrics.functional.regression import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    spearman_corrcoef,
    kendall_rank_corrcoef
)
from torchmetrics.retrieval import RetrievalMRR, RetrievalNormalizedDCG
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EvalPrediction,
    TrainerCallback,
)
import wandb
from config import WANDB_ENTITY, DEBERTA_MAX_LENGTH
from .modules import RankNetLoss, ListNetLoss, LambdaRankLoss, ListMLELoss

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
        # self.batch_size = batch_size

        groups = defaultdict(list)
        for idx, item in enumerate(dataset):
            question_id = item["labels"]["question_id"]
            if torch.is_tensor(question_id):
                question_id = question_id.item()
            groups[question_id].append(idx)

        self.groups = groups.values()

    def __iter__(self):
        # batch = []
        # batch_len = 0

        # for group in self.groups:
        #     random.shuffle(group)
        #     group_size = len(group)
        #     if batch_len + group_size > self.batch_size and batch:
        #         yield batch
        #         batch = []
        #         batch_len = 0
        #     batch.extend(group)
        #     batch_len += group_size

        # if batch:
        #     yield batch

        yield from self.groups

    def __len__(self):
        # count = 0
        # batch_len = 0

        # for group in self.groups:
        #     group_size = len(group)
        #     if batch_len + group_size > self.batch_size and batch_len > 0:
        #         count += 1
        #         batch_len = 0
        #     batch_len += group_size

        # if batch_len > 0:
        #     count += 1

        # return count

        return len(self.groups)


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
        self.loss = kwargs.pop("loss", "mse")
        margin = kwargs.pop("margin", 0.1)
        sigma = kwargs.pop("sigma", 1.0)

        if self.loss == "mse":
            self.loss_fn = nn.MSELoss()
        elif self.loss == "margin":
            self.loss_fn = nn.MarginRankingLoss(margin=margin)
        elif self.loss == "ranknet":
            self.loss_fn = RankNetLoss(sigma=sigma)
        elif self.loss == "listnet":
            self.loss_fn = ListNetLoss()
        elif self.loss == "lambdarank":
            self.loss_fn = LambdaRankLoss(sigma=sigma)
        elif self.loss == "listmle":
            self.loss_fn = ListMLELoss()
        else:
            raise ValueError(f"Unknown loss function: {self.loss}")

        super().__init__(*args, **kwargs)

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
        indexes = labels_dict["question_ids"]

        if self.loss == "mse":
            loss = self.loss_fn(predictions, labels)
        elif self.loss == "margin":
            labels_diff = labels.unsqueeze(0) - labels.unsqueeze(1)
            mask = labels_diff > 0
            if mask.any():
                i_idx, j_idx = mask.nonzero(as_tuple=True)
                pred_i = predictions[i_idx]
                pred_j = predictions[j_idx]
                target = torch.ones_like(pred_i)
                loss = self.loss_fn(pred_i, pred_j, target)
            else:
                loss = torch.tensor(0.0, device=predictions.device)
        elif self.loss in ["ranknet", "listnet", "lambdarank", "listmle"]:
            loss = self.loss_fn(predictions, labels, indexes)
        else:
            raise ValueError(f"Unknown loss function: {self.loss}")

        if return_outputs:
            return (loss, outputs)
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
    parser.add_argument("--learning-rate", help="Learning Rate", type=float, default=2e-5)
    parser.add_argument("--lr-scheduler-type", help="Learning Rate Scheduler Type", type=str, default="cosine")
    parser.add_argument("--warmup-ratio", help="Warmup Ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", help="Weight Decay", type=float, default=0.01)
    parser.add_argument("--batch-size", help="Batch Size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", help="Gradient Accumulation Steps", type=int, default=4)
    parser.add_argument("--num-epochs", help="Number of Epochs", type=int, default=3)
    parser.add_argument("--fp16", help="Use FP16", action="store_true")
    parser.add_argument("--loss", help="Loss Function", type=str, default="mse", choices=['mse', 'margin', 'ranknet', 'listnet', 'lambdarank', 'listmle'])
    parser.add_argument("--margin", help="Margin for Ranking Loss", type=float, default=0.1)
    parser.add_argument("--sigma", help="Sigma for RankNet or LambdaRank", type=float, default=1.0)

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
                "loss": args.loss,
                "margin": args.margin,
                "sigma": args.sigma,
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
        ddp_find_unused_parameters=False,
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
        loss=args.loss,
        margin=args.margin,
        sigma=args.sigma,
    )

    trainer.train()

    print("Training completed. Evaluating on test dataset...\n")

    test_dataset = VerifierDataset(args.test_data_path, tokenizer, args.max_length, args.desired_scale)
    _, _, metrics = trainer.predict(test_dataset)

    print("Test Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    wandb.finish()


if __name__ == "__main__":
    main()
