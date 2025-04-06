import json
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import Llama3_8b_PATH 

def gen_retriever_query_prompt(question, confirmed_passages):
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "Your task is to generate a retrieval query that targets the missing or complementary evidence not covered in the provided passages.\n"
        "Steps:\n"
        "1. Read the question.\n"
        "2. Analyze the provided passages.\n"
        "3. Identify key details missing from them.\n"
        "Output only the retrieval query—do not include any additional explanation or commentary.\n"
        "<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"Question: {question}\n"
    )
    if confirmed_passages:
        for idx, passage in enumerate(confirmed_passages, start=1):
            prompt += f"Confirmed Passage {idx}: {passage}\n"
    prompt += "<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
    return prompt.strip()


def load_retrieval_data(jsonl_path):
    records = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            qid = entry["question_id"]
            question = entry["question"]
            passages = entry["ctxs"]
            for passage in passages:
                passage_text = passage["text"]
                records.append((qid, question, passage_text))
    return records

def load_predictions(pred_path):
    predictions = []
    with open(pred_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            qid, passage_id, score_str = parts[0], parts[1], parts[2]
            try:
                score = float(score_str)
            except Exception:
                score = 0.0
            predictions.append((qid, passage_id, score))
    return predictions

def main():
    parser = argparse.ArgumentParser(
        description="Generate retriever queries using the best passage (based on verifier scores) for each question."
    )
    parser.add_argument("--test_data_path", type=str, required=True,
                        help="Path to the JSONL file containing retrieval results (question_id, question, ctxs)")
    parser.add_argument("--predictions_path", type=str, required=True,
                        help="Path to the predictions.txt file containing question_id, passage_id, and score for each passage")
    parser.add_argument("--model_id", type=str, required=True,
                        help="Model id for the causal LM (e.g. meta-llama/Meta-Llama-3-8B-Instruct)")
    parser.add_argument("--max_gen_length", type=int, default=50,
                        help="Maximum additional length for generation (default: 50)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for generation (default: 0.7)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p for generation (default: 0.9)")
    args = parser.parse_args()

    retrieval_records = load_retrieval_data(args.test_data_path)
    pred_list = load_predictions(args.predictions_path)
    
    if len(retrieval_records) != len(pred_list):
        print("Warning: Number of passages in retrieval data and predictions do not match!")
    
    records_with_score = []
    for (qid, question, passage_text), (pred_qid, pred_passage_id, score) in zip(retrieval_records, pred_list):
        records_with_score.append((qid, question, passage_text, score))
    
    best_records = dict() 
    for qid, question, passage_text, score in records_with_score:
        if qid not in best_records or score > best_records[qid][2]:
            best_records[qid] = (question, passage_text, score)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, cache_dir=Llama3_8b_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        cache_dir=Llama3_8b_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    
    for qid, (question, best_passage, score) in best_records.items():
        prompt = gen_retriever_query_prompt(question, [best_passage])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_length=inputs['input_ids'].shape[1] + args.max_gen_length,
            temperature=args.temperature,
            top_p=args.top_p
        )
        generated_text = tokenizer.decode(
            outputs[0][inputs['input_ids'].size(1):],
            skip_special_tokens=True
        ).strip()
        
        print(f"Question ID: {qid}")
        print("Best Verifier Score:", score)
        print("Generated Retriever Query:", generated_text)
        print("=" * 80)

if __name__ == "__main__":
    main()
