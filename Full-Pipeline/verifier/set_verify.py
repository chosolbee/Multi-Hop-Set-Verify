import os
import json
import torch
import argparse
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EvalPrediction,
)

DEFAULT_MAX_LENGTH = 768

class VerifierDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_length=DEFAULT_MAX_LENGTH, prev_context=""):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        self.example_ids = []  
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                qid = entry.get("question_id", None)
                question = entry["question"]
                if prev_context:
                    prefix = f"Question: {question} [SEP] {prev_context} [SEP] "
                else:
                    prefix = f"Question: {question} [SEP] "
                    
                passages = entry["ctxs"]
                for passage in passages:
                    passage_text = passage["text"]
                    text = prefix + passage_text
                    self.data.append({"text": text})
                    self.example_ids.append((qid, passage.get("id", None)))
    
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
        return encoding

def collate_fn(batch):
    input_ids = torch.stack([x["input_ids"] for x in batch])
    attention_mask = torch.stack([x["attention_mask"] for x in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask}

def compute_metrics(eval_pred: EvalPrediction):
    predictions = eval_pred.predictions.flatten()
    return {"dummy_metric": 0.0}

def main():
    parser = argparse.ArgumentParser(description="Verifier Inference with retrieval results")
    parser.add_argument("--test_data_path", type=str, required=True,
                        help="Path to the JSONL file containing retrieval results")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model name or path for the tokenizer")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the model checkpoint")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save prediction results")
    parser.add_argument("--batch_size", type=int, required=True,
                        help="Batch size for inference")
    parser.add_argument("--max_length", type=int, required=False, default=DEFAULT_MAX_LENGTH,
                        help="Maximum sequence length for tokenizer (default: 768)")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint_path, num_labels=1)
    model.config.problem_type = "regression"

    test_dataset = VerifierDataset(args.test_data_path, tokenizer, max_length=args.max_length)
    example_ids = test_dataset.example_ids 

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_eval_batch_size=args.batch_size,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    predictions_output = trainer.predict(test_dataset)
    preds = predictions_output.predictions.flatten()
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "predictions.txt")
    with open(output_file, "w") as fout:
        for (qid, passage_id), score in zip(example_ids, preds.tolist()):
            fout.write(f"{qid}\t{passage_id}\t{score:.4f}\n")
    print(f"Saved predictions to {output_file}")

if __name__ == "__main__":
    main()
