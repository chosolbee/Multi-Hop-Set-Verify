import os
import time
import json
import argparse
import wandb
from config import WANDB_ENTITY
from .contriever import Retriever
from .query_generator.query_generator_wo_verifier import QueryGenerator

def print_results(em_list, precision_list, recall_list, f1_list):
    em_flat = [item for sublist in em_list for item in sublist]
    precision_flat = [item for sublist in precision_list for item in sublist]
    recall_flat = [item for sublist in recall_list for item in sublist]
    f1_flat = [item for sublist in f1_list for item in sublist]

    print("Count:", [len(em) for em in em_list])
    print("EM:", [sum(em) / len(em) if em else 0 for em in em_list])
    print("Total EM", sum(em_flat) / len(em_flat) if em_flat else 0)
    print("Precision:", [sum(pr) / len(pr) if pr else 0 for pr in precision_list])
    print("Total Precision", sum(precision_flat) / len(precision_flat) if precision_flat else 0)
    print("Recall:", [sum(rc) / len(rc) if rc else 0 for rc in recall_list])
    print("Total Recall", sum(recall_flat) / len(recall_flat) if recall_flat else 0)
    print("F1:", [sum(f1) / len(f1) if f1 else 0 for f1 in f1_list])
    print("Total F1", sum(f1_flat) / len(f1_flat) if f1_flat else 0)
    print()


def run_batch(retriever, query_generator, questions,
              max_iterations=5, max_search=10, log_trace=False,
              stop_log_path=None):
    final_questions = []
    final_batch_history = []
    batch_history = [[] for _ in questions]
    iter_count = 0

    stop_logs = []

    while questions:
        start_time = time.time()

        queries = query_generator.batch_generate(questions, batch_history)
        batch_docs = retriever.search(queries, max_search)

        if log_trace:
            for question, query, history, docs in zip(questions, queries, batch_history, batch_docs):
                print(f"1. Question: {question['question']}")
                print("2. History:")
                for doc in history:
                    print(f"  Passage: {doc['text']}")
                print(f"3. Generated query: {query}")
                print("4. Retrieved Passages:")
                for doc in docs:
                    print(f"  Passage: {doc['text']}")
                print()

        next_questions = []
        next_batch_history = []

        for question, history, query, docs in zip(questions, batch_history, queries, batch_docs):
            if query.strip().lower() == "<stop>":
                final_questions.append(question)
                final_batch_history.append(history)

                stop_logs.append({
                    "question_id": question["id"],
                    "gold_hop": len(question.get("question_decomposition", [])),
                    "stop_iter": iter_count + 1
                })
            else:
                existing_ids = {d["id"] for d in history}
                for cand in docs:
                    if cand["id"] not in existing_ids:
                        history.append(cand)
                        break
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
                stop_logs.append({
                    "question_id": question["id"],
                    "gold_hop": len(question.get("question_decomposition", [])),
                    "stop_iter": iter_count
                })
            break

    em_list = [[], [], []]
    precision_list = [[], [], []]
    recall_list = [[], [], []]
    f1_list = [[], [], []]

    for question, history in zip(final_questions, final_batch_history):
        qid = question["id"]
        decomposition = question.get("question_decomposition", [])
        gold_idxs = [step.get("paragraph_support_idx") for step in decomposition]
        gold_chunk_ids = {f"{qid}-{idx:02d}" for idx in gold_idxs if idx is not None}
        gold_hop = len(gold_chunk_ids)

        correct = sum(1 for doc in history if gold_chunk_ids & set(doc["id"].split("//")))
        retrieved = len(history)
        em = int(correct == gold_hop and retrieved == gold_hop)
        precision = correct / retrieved if retrieved else 0.0
        recall = correct / gold_hop if gold_hop else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        for log in stop_logs:
            if log.get("question_id") == qid:
                log.update({"em": em, "precision": precision, "recall": recall, "f1": f1})
                break

        idx = max(gold_hop - 2, 0)
        em_list[idx].append(em)
        precision_list[idx].append(precision)
        recall_list[idx].append(recall)
        f1_list[idx].append(f1)

    if stop_log_path:
        with open(stop_log_path, "a", encoding="utf-8") as f:
            for log in stop_logs:
                f.write(json.dumps(log, ensure_ascii=False) + "\n")

    return em_list, precision_list, recall_list, f1_list


def parse_args():
    parser = argparse.ArgumentParser()

    retriever_group = parser.add_argument_group("Retriever Options")
    retriever_group.add_argument("--passages", type=str, required=True, help="document file path")
    retriever_group.add_argument("--embeddings", type=str, required=True, help="Document embedding path")

    query_generator_group = parser.add_argument_group("Query Generator Options")
    query_generator_group.add_argument("--qg-model-id", type=str, default="meta-llama/Llama-3.1-8B-instruct", help="Model ID for query generator")
    query_generator_group.add_argument("--qg-tp-size", type=int, default=1, help="Tensor parallel size for query generator")
    query_generator_group.add_argument("--qg-quantization", type=str, help="Quantization method for query generator")
    query_generator_group.add_argument("--qg-max-gen-length", type=int, default=512, help="Maximum generation length for query generator")
    query_generator_group.add_argument("--qg-temperature", type=float, default=0.7, help="Temperature for query generator")
    query_generator_group.add_argument("--qg-top-p", type=float, default=0.9, help="Top-p sampling for query generator")

    main_group = parser.add_argument_group("Main Options")
    main_group.add_argument("--questions", type=str, required=True, help="Questions file path")
    main_group.add_argument("--batch-size", type=int, default=32, help="Batch size for processing questions")
    main_group.add_argument("--max-iterations", type=int, default=5, help="Maximum number of iterations")
    main_group.add_argument("--max-search", type=int, default=10, help="Maximum number of passages to retrieve")
    main_group.add_argument("--log-trace", action="store_true", help="Log trace for debugging")
    main_group.add_argument("--stop-log-path", type=str, default=None, help="Optional JSONL path; Path to the JSONL file where stopping logs are written")

    return parser.parse_args()


def main(args: argparse.Namespace):
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

    query_generator = QueryGenerator(
        model_id=args.qg_model_id,
        tp_size=args.qg_tp_size,
        quantization=args.qg_quantization,
        max_gen_length=args.qg_max_gen_length,
        temperature=args.qg_temperature,
        top_p=args.qg_top_p,
    )

    if args.stop_log_path:
        open(args.stop_log_path, "w", encoding="utf-8").close()

    with open(args.questions, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]

    em_list = [[], [], []]
    precision_list = [[], [], []]
    recall_list = [[], [], []]
    f1_list = [[], [], []]

    for i in range(0, len(questions), args.batch_size):
        batch_questions = questions[i: i + args.batch_size]
        print(f"Processing batch {i // args.batch_size + 1} of {(len(questions)-1) // args.batch_size + 1}...\n")

        em, precision, recall, f1 = run_batch(
            retriever=retriever,
            query_generator=query_generator,
            questions=batch_questions,
            max_iterations=args.max_iterations,
            max_search=args.max_search,
            log_trace=args.log_trace,
            stop_log_path=args.stop_log_path,
        )
        for j in range(3):
            em_list[j].extend(em[j])
            precision_list[j].extend(precision[j])
            recall_list[j].extend(recall[j])
            f1_list[j].extend(f1[j])
        print_results(em_list, precision_list, recall_list, f1_list)

    print("Final Results:")
    print_results(em_list, precision_list, recall_list, f1_list)
    print("All done!")

    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)
