import os
import json
import argparse
import torch
import numpy as np
from torch.utils.data import Dataset, random_split, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EvalPrediction,
    TrainerCallback,
)
from config import WANDB_ENTITY
import torch.nn as nn
import wandb

class VerifierDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_length=768, desired_scale=10):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.desired_scale = desired_scale
        self.data = []
        raw_scores = []

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                question = entry["question"]
                passages = entry["passages"]
                passages_text = " [SEP] ".join([p["paragraph_text"] for p in passages])
                text = f"Question: {question} [SEP] {passages_text}"
                score = entry["score"]
                raw_scores.append(score)
                self.data.append({"text": text, "raw_score": score})

        self.min_score = min(raw_scores)
        self.max_score = max(raw_scores)
        if self.max_score == self.min_score:
            self.max_score += 1e-6
        for item in self.data:
            normalized = (item["raw_score"] - self.min_score) / (self.max_score - self.min_score)
            item["score"] = normalized * self.desired_scale

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
        encoding["labels"] = torch.tensor(item["score"], dtype=torch.float)
        return encoding

def collate_fn(batch):
    input_ids = torch.stack([x["input_ids"] for x in batch])
    attention_mask = torch.stack([x["attention_mask"] for x in batch])
    labels = torch.stack([x["labels"] for x in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def compute_metrics(eval_pred: EvalPrediction):
    predictions, labels = eval_pred
    predictions = predictions.flatten()
    mse = ((predictions - labels.flatten()) ** 2).mean().item()
    return {"mse": mse}

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        predictions = outputs.logits.squeeze(-1)
        labels = inputs["labels"]
        mse_loss = nn.MSELoss()(predictions, labels)
        
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
    parser = argparse.ArgumentParser(
        description="Verifier Train",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("-t", "--train", help="Training Set Path", type=str)
    parser.add_argument("-v", "--val", help="Evaluation Set Path", type=str)
    parser.add_argument("-w", "--wandb_entity", default=0.5, help="Lambda Value", type=float)
    
    return parser.parse_args()

def main():
    parser = argparse.ArgumentParser(description="Verifier Train")
    parser.add_argument("-t", "--train", help="Training Set Path", type=str)
    parser.add_argument("-v", "--val", help="Evaluation Set Path", type=str)
    args = parser.parse_args()

    wandb.init(project="MultiHopVerifierTraining", entity=WANDB_ENTITY)
    train_data_path = args.train
    val_data_path = args.val
    model_name = "microsoft/deberta-v3-large"
    max_length = 768
    desired_scale = 10
    batch_size = 8
    num_epochs = 3
    learning_rate = 5e-5

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1, problem_type="regression")

    train_dataset = VerifierDataset(train_data_path, tokenizer, max_length=max_length, desired_scale=desired_scale)
    val_dataset = VerifierDataset(val_data_path, tokenizer, max_length=max_length, desired_scale=desired_scale)
    training_args = TrainingArguments(
        output_dir="./verifier_results",
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=num_epochs,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=500,
        logging_steps=100,
        weight_decay=0.01,
        report_to=["wandb"],
        load_best_model_at_end=True,
        fp16=True,
    )
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        callbacks=[PrintCallback()]
    )
    trainer.train()
    wandb.finish()

if __name__ == "__main__":
    main()
