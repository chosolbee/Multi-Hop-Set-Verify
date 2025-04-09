import os
import json
import argparse
import torch
import numpy as np
from torch.utils.data import Dataset, random_split, DataLoader
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError, R2Score
from torchmetrics.retrieval import RetrievalMAP, RetrievalMRR, RetrievalNormalizedDCG
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


class VerifierDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_length=768, desired_scale=1):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.desired_scale = desired_scale
        self.data = []
        question_ids = []

        with open(filepath, 'r', encoding='utf-8') as f:
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


def collate_fn(batch):
    input_ids = torch.stack([x["input_ids"] for x in batch])
    attention_mask = torch.stack([x["attention_mask"] for x in batch])
    labels = torch.stack([x["labels"]["labels"] for x in batch])
    question_ids = torch.stack([x["labels"]["question_id"] for x in batch])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": {
            "labels": labels,
            "question_ids": question_ids,
        }
    }


class Metrics:
    def __init__(self, desired_scale=1):
        self.desired_scale = desired_scale
        self.mse_metric = MeanSquaredError()
        self.mae_metric = MeanAbsoluteError()
        self.r2_metric = R2Score()
        self.map_metric = RetrievalMAP()
        self.mrr_metric = RetrievalMRR()
        self.ndcg_metric = RetrievalNormalizedDCG()


    def __call__(self, eval_preds: EvalPrediction):
        logits, labels_dict = eval_preds
        logits = torch.tensor(logits.flatten())
        labels = torch.tensor(labels_dict["labels"].flatten())
        indexes = torch.tensor(labels_dict["question_ids"].flatten())

        predictions = torch.sigmoid(logits)
        labels = labels / self.desired_scale

        return {
            "mse": self.mse_metric(predictions, labels).item(),
            "mae": self.mae_metric(predictions, labels).item(),
            "r2": self.r2_metric(predictions, labels).item(),
            "map": self.map_metric(predictions, labels, indexes).item(),
            "mrr": self.mrr_metric(predictions, labels, indexes).item(),
            "ndcg": self.ndcg_metric(predictions, labels, indexes).item(),
        }


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.desired_scale = kwargs.pop("desired_scale", 1)
        super().__init__(*args, **kwargs)

        self.mse_loss = nn.MSELoss()


    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels_dict = inputs.pop("labels", None)
        print(labels_dict)

        outputs = model(**inputs)

        predictions = torch.sigmoid(outputs.logits.squeeze(-1)) * self.desired_scale
        labels = labels_dict["labels"]
        question_ids = labels_dict["question_ids"]

        mse_loss = self.mse_loss(predictions, labels)
        
        # margin = 1.0
        # candidate_pairs = []
        # for i in range(len(labels)):
        #     for j in range(len(labels)):
        #         if labels[i] > labels[j]:
        #             candidate_pairs.append((i, j))
        # if candidate_pairs:
        #     pred_i = torch.stack([predictions[i] for i, j in candidate_pairs])
        #     pred_j = torch.stack([predictions[j] for i, j in candidate_pairs])
        #     target = torch.ones_like(pred_i)
        #     margin_loss = nn.MarginRankingLoss(margin=margin)(pred_i, pred_j, target)
        # else:
        #     margin_loss = torch.tensor(0.0, device=predictions.device)
        # alpha = 0.5
        # loss = mse_loss + alpha * margin_loss
        
        loss = mse_loss
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
    
    parser.add_argument("-t", "--train-data-path", help="Training Dataset Path", type=str)
    parser.add_argument("-v", "--eval-data-path", help="Evaluation Dataset Path", type=str)
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
            }
        )
    else:
        os.environ["WANDB_MODE"] = "disabled"  # Disable wandb for other processes

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1, problem_type="regression")

    train_dataset = VerifierDataset(args.train_data_path, tokenizer, max_length=args.max_length, desired_scale=args.desired_scale)
    eval_dataset = VerifierDataset(args.eval_data_path, tokenizer, max_length=args.max_length, desired_scale=args.desired_scale)
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
        eval_steps=5,
        save_steps=500,
        logging_steps=100,
        report_to=["wandb"],
        load_best_model_at_end=True,
        fp16=args.fp16,
    )

    compute_metrics = Metrics(desired_scale=args.desired_scale)
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        callbacks=[PrintCallback()],
        desired_scale=args.desired_scale,
    )
    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    main()
