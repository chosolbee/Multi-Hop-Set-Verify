import os
import re
import time
import json
import argparse
from collections import Counter
from typing import List, Dict, Any, Tuple
import torch
from vllm import LLM
import wandb

from config import WANDB_ENTITY, DEBERTA_MAX_LENGTH
from .contriever import Retriever
from .query_generator import QueryGenerator
from .verifier import Verifier
from .answer_generator import AnswerGenerator


def print_metrics(metrics_list, metric_name="Metrics"):
    if not metrics_list or not any(metrics_list):
        print(f"No {metric_name} available.")
        return

    metrics_flat = [item for sublist in metrics_list for item in sublist]

    print(f"===== {metric_name} =====")
    print(f"Count: {[len(m) for m in metrics_list]}")

    averages = [sum(m) / len(m) if m else 0 for m in metrics_list]
    print(f"{metric_name} by hop: {[f'{avg:.4f}' for avg in averages]}")

    total_avg = sum(metrics_flat) / len(metrics_flat) if metrics_flat else 0
    print(f"Total {metric_name}: {total_avg:.4f}")


def normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r'\b(a|an|the)\b', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return 0.0

    common = Counter(p) & Counter(g)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(p)
    recall = overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def compute_retrieval_metrics(questions: List[Dict[str, Any]],
                              histories: List[List[Dict[str, Any]]]) -> Tuple[List[List[float]], ...]:
    em_list = [[], [], []]
    precision_list = [[], [], []]
    recall_list = [[], [], []]
    f1_list = [[], [], []]

    for question, history in zip(questions, histories):
        qid = question["id"]
        gold_hop = len(question.get("question_decomposition", []))
        correct = sum(int(qid + "-sf" in doc["id"]) for doc in history)
        retrieved = len(history)

        em = int(correct == gold_hop and retrieved == gold_hop)
        precision = correct / retrieved if retrieved else 0.0
        recall = correct / gold_hop if gold_hop else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        idx = min(max(gold_hop - 2, 0), 2)
        em_list[idx].append(em)
        precision_list[idx].append(precision)
        recall_list[idx].append(recall)
        f1_list[idx].append(f1)

    return em_list, precision_list, recall_list, f1_list


def compute_answer_metrics(questions: List[Dict[str, Any]],
                          predictions: List[str]) -> Tuple[List[List[float]], List[List[float]]]:
    em_list = [[], [], []]
    f1_list = [[], [], []]

    for question, prediction in zip(questions, predictions):
        hop = len(question.get("question_decomposition", []))
        idx = min(max(hop - 2, 0), 2)

        gold_answers = [question["answer"]] + question.get("answer_aliases", [])
        gold_answers = [normalize(g) for g in gold_answers]

        norm_pred = normalize(prediction)

        em = int(norm_pred in gold_answers)
        f1 = max(token_f1(norm_pred, g) for g in gold_answers)

        em_list[idx].append(em)
        f1_list[idx].append(f1)

    return em_list, f1_list


def run_batch(retriever: Retriever,
              query_generator: QueryGenerator,
              verifier: Verifier,
              answer_generator: AnswerGenerator,
              questions: List[Dict[str, Any]],
              max_iterations: int = 5,
              max_search: int = 10,
              verifier_threshold: float = 0.9,
              log_trace: bool = False,
              generate_answers: bool = False,
              stop_log_path: str = None) -> Tuple[List[Dict[str, Any]], List[List[Dict[str, Any]]], Dict[str, List[List[float]]], List[str]]:

    final_questions = []
    final_batch_history = []
    batch_history = [[] for _ in range(len(questions))]
    iter_count = 0
    stop_logs = []

    while questions:
        start_time = time.time()

        queries = query_generator.batch_generate(questions, batch_history)
        batch_docs = retriever.search(queries, max_search)
        batch_scores = verifier.batch_verify(questions, batch_history, batch_docs)

        if log_trace:
            for question, query, history, docs, scores in zip(questions, queries, batch_history, batch_docs, batch_scores):
                print(f"1. Question: {question['question']}")
                print("2. History:")
                for doc in history:
                    print(f"  Passage: {doc['text']}")
                print(f"3. Generated query: {query}")
                print("4. Retrieved passages and scores:")
                for doc, score in zip(docs, scores):
                    print(f"  Score: {score:.2f} | Passage: {doc['text']}")
                print()

        next_questions = []
        next_batch_history = []

        for question, history, docs, scores in zip(questions, batch_history, batch_docs, batch_scores):
            for i, doc in enumerate(docs):
                if doc["id"] in {d["id"] for d in history}:
                    scores[i] = -1

            max_score = scores.max()
            max_idx = scores.argmax()
            history.append(docs[max_idx])

            if max_score > verifier_threshold:
                final_questions.append(question)
                final_batch_history.append(history)

                gold_hop = len(question.get("question_decomposition", []))
                stop_logs.append({
                    "question_id": question["id"],
                    "gold_hop": gold_hop,
                    "stop_iter": iter_count + 1
                })
            else:
                next_questions.append(question)
                next_batch_history.append(history)

        questions = next_questions
        batch_history = next_batch_history

        print(f"Iteration {iter_count+1} completed in {time.time() - start_time:.2f} seconds")
        print(f"Remaining questions: {len(questions)}\n")

        iter_count += 1
        if iter_count >= max_iterations:
            for question, history in zip(questions, batch_history):
                final_questions.append(question)
                final_batch_history.append(history)
                gold_hop = len(question.get("question_decomposition", []))
                stop_logs.append({
                    "question_id": question["id"],
                    "gold_hop": gold_hop,
                    "stop_iter": iter_count
                })
            break

    if log_trace:
        print("\nFinal Questions and History:\n")
        for question, history in zip(final_questions, final_batch_history):
            print(f"1. Question: {question['question']}")
            print("2. History:")
            for doc in history:
                print(f"  Passage: {doc['text']}")
            print()

    em_list, precision_list, recall_list, f1_list = compute_retrieval_metrics(final_questions, final_batch_history)

    for question, history in zip(final_questions, final_batch_history):
        qid = question["id"]
        gold_hop = len(question.get("question_decomposition", []))
        correct = sum(int(qid + "-sf" in doc["id"]) for doc in history)
        retrieved = len(history)

        em = int(correct == gold_hop and retrieved == gold_hop)
        precision = correct / retrieved if retrieved else 0.0
        recall = correct / gold_hop if gold_hop else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        for log in stop_logs:
            if log["question_id"] == qid:
                log.update({"em": em, "precision": precision, "recall": recall, "f1": f1})
                break

    if stop_log_path:
        with open(stop_log_path, 'a', encoding='utf-8') as f:
            for log in stop_logs:
                f.write(json.dumps(log, ensure_ascii=False) + '\n')

    predictions = []
    ans_em_list = [[], [], []]
    ans_f1_list = [[], [], []]

    if generate_answers and answer_generator:
        predictions = answer_generator.batch_answer(final_questions, final_batch_history)
        ans_em_list, ans_f1_list = compute_answer_metrics(final_questions, predictions)

    metrics = {
        "retrieval": {
            "em": em_list,
            "precision": precision_list,
            "recall": recall_list,
            "f1": f1_list
        },
        "answer": {
            "em": ans_em_list,
            "f1": ans_f1_list
        }
    }

    return final_questions, final_batch_history, metrics, predictions


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-hop Question Answering Pipeline")

    retriever_group = parser.add_argument_group("Retriever Options")
    retriever_group.add_argument("--passages", type=str, required=True, help="Document file path")
    retriever_group.add_argument("--embeddings", type=str, required=True, help="Document embedding path")

    query_generator_group = parser.add_argument_group("Query Generator Options")
    query_generator_group.add_argument("--qg-model-id", type=str, default="meta-llama/Llama-3.1-8B-instruct", help="Model ID for query generator")
    query_generator_group.add_argument("--qg-tp-size", type=int, default=1, help="Tensor parallel size for query generator")
    query_generator_group.add_argument("--qg-quantization", type=str, help="Quantization method for query generator")
    query_generator_group.add_argument("--qg-max-gen-length", type=int, default=512, help="Maximum generation length for query generator")
    query_generator_group.add_argument("--qg-temperature", type=float, default=0.7, help="Temperature for query generator")
    query_generator_group.add_argument("--qg-top-p", type=float, default=0.9, help="Top-p sampling for query generator")

    verifier_group = parser.add_argument_group("Verifier Options")
    verifier_group.add_argument("--verifier-model-id", type=str, default="microsoft/DeBERTa-v3-large", help="Model ID for verifier")
    verifier_group.add_argument("--verifier-checkpoint-path", type=str, required=True, help="Checkpoint path for trained model")
    verifier_group.add_argument("--verifier-batch-size", type=int, default=8, help="Batch size for verifier")
    verifier_group.add_argument("--verifier-max-length", type=int, default=DEBERTA_MAX_LENGTH, help="Maximum length for verifier input")

    answer_generator_group = parser.add_argument_group("Answer Generator Options")
    answer_generator_group.add_argument("--generate-answers", action="store_true", help="Enable answer generation")
    answer_generator_group.add_argument("--ag-max-gen-length", type=int, default=1024, help="Maximum generation length for answer generator")
    answer_generator_group.add_argument("--ag-temperature", type=float, default=0.7, help="Temperature for answer generator")
    answer_generator_group.add_argument("--ag-top-p", type=float, default=0.9, help="Top-p sampling for answer generator")

    main_group = parser.add_argument_group("Main Options")
    main_group.add_argument("--questions", type=str, required=True, help="Questions file path")
    main_group.add_argument("--batch-size", type=int, default=32, help="Batch size for processing questions")
    main_group.add_argument("--max-iterations", type=int, default=5, help="Maximum number of iterations")
    main_group.add_argument("--max-search", type=int, default=10, help="Maximum number of passages to retrieve")
    main_group.add_argument("--verifier-threshold", type=float, default=0.9, help="Threshold for verifier scores")
    main_group.add_argument("--log-trace", action="store_true", help="Log trace for debugging")
    main_group.add_argument("--output-path", type=str, help="Path to save predictions and metrics")
    main_group.add_argument("--stop-log-path", type=str, default=None, help="Optional JSONL path for stopping logs")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank in [-1, 0]:
        wandb.init(project="MultiHopQA-test", entity=WANDB_ENTITY)
    else:
        os.environ["WANDB_MODE"] = "disabled"

    retriever = Retriever(
        args.passages,
        args.embeddings,
        model_type="contriever",
        model_path="facebook/contriever-msmarco",
    )

    shared_llm = LLM(
        model=args.qg_model_id,
        tensor_parallel_size=args.qg_tp_size,
        quantization=args.qg_quantization,
        dtype=torch.bfloat16,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
    )

    query_generator = QueryGenerator(
        llm=shared_llm,
        max_gen_length=args.qg_max_gen_length,
        temperature=args.qg_temperature,
        top_p=args.qg_top_p,
    )

    verifier = Verifier(
        model_id=args.verifier_model_id,
        checkpoint_path=args.verifier_checkpoint_path,
        batch_size=args.verifier_batch_size,
        max_length=args.verifier_max_length,
    )

    answer_generator = None
    if args.generate_answers:
        answer_generator = AnswerGenerator(
            llm=shared_llm,
            max_gen_length=args.ag_max_gen_length,
            temperature=args.ag_temperature,
            top_p=args.ag_top_p,
        )

    if args.stop_log_path:
        open(args.stop_log_path, "w", encoding="utf-8").close()

    with open(args.questions, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]

    all_metrics = {
        "retrieval": {
            "em": [[], [], []],
            "precision": [[], [], []],
            "recall": [[], [], []],
            "f1": [[], [], []]
        },
        "answer": {
            "em": [[], [], []],
            "f1": [[], [], []]
        }
    }

    all_predictions = []
    all_final_questions = []
    all_final_histories = []

    total_batches = (len(questions) + args.batch_size - 1) // args.batch_size
    for i in range(0, len(questions), args.batch_size):
        batch_questions = questions[i:i + args.batch_size]
        print(f"\nProcessing batch {i // args.batch_size + 1} of {total_batches}...\n")

        final_questions, final_histories, batch_metrics, predictions = run_batch(
            retriever=retriever,
            query_generator=query_generator,
            verifier=verifier,
            answer_generator=answer_generator,
            questions=batch_questions,
            max_iterations=args.max_iterations,
            max_search=args.max_search,
            verifier_threshold=args.verifier_threshold,
            log_trace=args.log_trace,
            generate_answers=args.generate_answers,
            stop_log_path=args.stop_log_path,
        )

        all_final_questions.extend(final_questions)
        all_final_histories.extend(final_histories)
        all_predictions.extend(predictions)

        for metric_type in ["retrieval", "answer"]:
            for metric_name in batch_metrics[metric_type]:
                for hop_idx in range(3):
                    all_metrics[metric_type][metric_name][hop_idx].extend(
                        batch_metrics[metric_type][metric_name][hop_idx]
                    )

        print("\n===== BATCH RETRIEVAL METRICS =====")
        print_metrics(batch_metrics["retrieval"]["em"], "EM")
        print_metrics(batch_metrics["retrieval"]["precision"], "Precision")
        print_metrics(batch_metrics["retrieval"]["recall"], "Recall")
        print_metrics(batch_metrics["retrieval"]["f1"], "F1")

        if args.generate_answers:
            print("\n===== BATCH ANSWER METRICS =====")
            print_metrics(batch_metrics["answer"]["em"], "EM")
            print_metrics(batch_metrics["answer"]["f1"], "F1")

    print("\n===== FINAL RETRIEVAL METRICS =====")
    print_metrics(all_metrics["retrieval"]["em"], "EM")
    print_metrics(all_metrics["retrieval"]["precision"], "Precision")
    print_metrics(all_metrics["retrieval"]["recall"], "Recall")
    print_metrics(all_metrics["retrieval"]["f1"], "F1")

    if args.generate_answers:
        print("\n===== FINAL ANSWER METRICS =====")
        print_metrics(all_metrics["answer"]["em"], "EM")
        print_metrics(all_metrics["answer"]["f1"], "F1")

    if args.output_path:
        output_data = {
            "metrics": all_metrics,
            "predictions": []
        }

        for q, h, p in zip(all_final_questions, all_final_histories, all_predictions):
            output_data["predictions"].append({
                "id": q["id"],
                "question": q["question"],
                "gold_answer": q.get("answer", ""),
                "prediction": p if p else "",
                "passages": [doc["text"] for doc in h]
            })

        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

    print("\nAll done!")
    wandb.finish()


if __name__ == "__main__":
    main()
