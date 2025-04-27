import os
import time
import json
import argparse
import wandb
from config import WANDB_ENTITY, DEBERTA_MAX_LENGTH
from .contriever import Retriever
from .query_generator import QueryGenerator
from .verifier import Verifier


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
    print("\n")


def run_batch(retriever, query_generator, verifier, questions,
              max_iterations=5, max_search=10, verifier_threshold=0.9,
              log_trace=False, stop_log_path=None):
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
                print("\n")

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

                stop_logs.append({
                    "question_id": question["id"],
                    "gold_hop": len(question.get("question_decomposition", [])),
                    "stop_iter": iter_count + 1
                })
            else:
                next_questions.append(question)
                next_batch_history.append(history)

        questions = next_questions
        batch_history = next_batch_history

        print("Iteration", iter_count + 1, "completed in", time.time() - start_time, "seconds")
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

    if log_trace:
        print("\nFinal Questions and History:\n")
        for question, history in zip(final_questions, final_batch_history):
            print(f"1. Question: {question['question']}")
            print("2. History:")
            for doc in history:
                print(f"  Passage: {doc['text']}")
            print("\n")

    em_list = [[], [], []]
    precision_list = [[], [], []]
    recall_list = [[], [], []]
    f1_list = [[], [], []]

    for question, history in zip(final_questions, final_batch_history):
        qid = question["id"]
        decomposition = question.get("question_decomposition", [])
        gold_idxs = [step["paragraph_support_idx"] for step in decomposition]
        gold_chunk_ids = {f"{qid}-{idx:02d}" for idx in gold_idxs}
        gold_hop = len(gold_chunk_ids)

        correct = sum(1 for doc in history if gold_chunk_ids & set(doc["id"].split("//")))
        retrieved = len(history)
        em = int(correct == gold_hop and retrieved == gold_hop)
        precision = correct / retrieved if retrieved else 0.0
        recall = correct / gold_hop if gold_hop else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        # DEBUG
        print(f"[DEBUG] qid={qid}, gold_hop={gold_hop}, history_ids={[d['id'] for d in history]}, "
              f"correct={correct}, retrieved={retrieved}, em={em:.3f}, recall={recall:.3f}, precision={precision:.3f}")

        for log in stop_logs:
            if log["question_id"] == qid:
                log.update({
                    "em": em,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1
                })
                break

        idx = gold_hop - 2
        em_list[idx].append(em)
        precision_list[idx].append(precision)
        recall_list[idx].append(recall)
        f1_list[idx].append(f1)

    if stop_log_path:
        with open(stop_log_path,'a',encoding='utf-8') as f:
            for log in stop_logs:
                f.write(json.dumps(log,ensure_ascii=False)+'\n')
                
    print_results(em_list, precision_list, recall_list, f1_list)
    return em_list, precision_list, recall_list, f1_list


def parse_args():
    parser = argparse.ArgumentParser()

    retriever_group = parser.add_argument_group("Retriever Options")
    retriever_group.add_argument("--passages", type=str, required=True, help="document file path")
    retriever_group.add_argument("--embeddings", type=str, required=True, help="Document embedding path")

    query_generator_group = parser.add_argument_group("Query Generator Options")
    query_generator_group.add_argument("--qg-cache-dir", type=str, help="Cache directory for query generator model")
    query_generator_group.add_argument("--qg-max-gen-length", type=int, default=512, help="Maximum generation length for query generator")
    query_generator_group.add_argument("--qg-temperature", type=float, default=0.7, help="Temperature for query generator")
    query_generator_group.add_argument("--qg-top-p", type=float, default=0.9, help="Top-p sampling for query generator")

    verifier_group = parser.add_argument_group("Verifier Options")
    verifier_group.add_argument("--verifier-checkpoint-path", type=str, required=True, help="Checkpoint path for trained model")
    verifier_group.add_argument("--verifier-batch-size", type=int, default=8, help="Batch size for verifier")
    verifier_group.add_argument("--verifier-max-length", type=int, default=DEBERTA_MAX_LENGTH, help="Maximum length for verifier input")

    main_group = parser.add_argument_group("Main Options")
    main_group.add_argument("--questions", type=str, required=True, help="Questions file path")
    main_group.add_argument("--batch-size", type=int, default=32, help="Batch size for processing questions")
    main_group.add_argument("--max-iterations", type=int, default=5, help="Maximum number of iterations")
    main_group.add_argument("--max-search", type=int, default=10, help="Maximum number of passages to retrieve")
    main_group.add_argument("--verifier-threshold", type=float, default=0.9, help="Threshold for verifier scores")
    main_group.add_argument("--log-trace", action="store_true", help="Log trace for debugging")
    main_group.add_argument("--stop-log-path", type=str, default=None,
                            help="Optional JSONL path; Path to the JSONL file where stopping logs are written")

    args = parser.parse_args()
    return args


def main(args: argparse.Namespace):
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank == -1 or local_rank == 0:
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
        model_id="meta-llama/Meta-Llama-3-8B-Instruct",
        cache_dir=args.qg_cache_dir,
        max_gen_length=args.qg_max_gen_length,
        temperature=args.qg_temperature,
        top_p=args.qg_top_p,
    )
    verifier = Verifier(
        model_id="microsoft/DeBERTa-v3-large",
        checkpoint_path=args.verifier_checkpoint_path,
        batch_size=args.verifier_batch_size,
        max_length=args.verifier_max_length,
    )

    if args.stop_log_path:
        open(args.stop_log_path,'w').close()

    with open(args.questions, "r") as f:
        questions = [json.loads(line) for line in f]

    em_list = [[], [], []]
    precision_list = [[], [], []]
    recall_list = [[], [], []]
    f1_list = [[], [], []]

    for i in range(0, len(questions), args.batch_size):
        batch_questions = questions[i: i + args.batch_size]
        print(f"Processing batch {i // args.batch_size + 1} of {len(questions) // args.batch_size + 1}...\n")
        em, precision, recall, f1 = run_batch(
            retriever=retriever,
            query_generator=query_generator,
            verifier=verifier,
            questions=batch_questions,
            max_iterations=args.max_iterations,
            max_search=args.max_search,
            verifier_threshold=args.verifier_threshold,
            log_trace=args.log_trace,
            stop_log_path=args.stop_log_path  
        )
        for j in range(3):
            em_list[j].extend(em[j])
            precision_list[j].extend(precision[j])
            recall_list[j].extend(recall[j])
            f1_list[j].extend(f1[j])

    print("Final Results:")
    print_results(em_list, precision_list, recall_list, f1_list)
    print("All done!")
    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)
