import os
import time
import json
import argparse
import wandb
from config import WANDB_ENTITY, DEBERTA_MAX_LENGTH
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
    print("Precision:", [sum(precision) / len(precision) if precision else 0 for precision in precision_list])
    print("Total Precision", sum(precision_flat) / len(precision_flat) if precision_flat else 0)
    print("Recall:", [sum(recall) / len(recall) if recall else 0 for recall in recall_list])
    print("Total Recall", sum(recall_flat) / len(recall_flat) if recall_flat else 0)
    print("F1:", [sum(f1) / len(f1) if f1 else 0 for f1 in f1_list])
    print("Total F1", sum(f1_flat) / len(f1_flat) if f1_flat else 0)
    print("\n")


def run_batch(retriever, query_generator, questions, max_iterations=5, max_search=1, log_trace=False):
    final_questions = []
    final_batch_history = []

    active_questions = questions[:]
    active_batch_history = [[] for _ in range(len(active_questions))]

    iter_count = 0
    while active_questions:
        start_time = time.time()

        queries = query_generator.batch_generate(active_questions, active_batch_history)

        new_active_questions = []
        new_active_batch_history = []
        remaining_queries = []
        for idx, query in enumerate(queries):
            if "<stop>" in query.strip().lower():
                print(f"Iteration {iter_count+1}: Question '{active_questions[idx]['question']}' reached stop condition with query: {query}")
                final_questions.append(active_questions[idx])
                final_batch_history.append(active_batch_history[idx])
            else:
                new_active_questions.append(active_questions[idx])
                new_active_batch_history.append(active_batch_history[idx])
                remaining_queries.append(query)
        active_questions = new_active_questions
        active_batch_history = new_active_batch_history

        if not active_questions:
            print("All questions reached stop condition. Exiting iterations.")
            break

        batch_docs = retriever.search(remaining_queries, max_search)

        for i, docs in enumerate(batch_docs):
            if docs:
                top_doc = docs[0]
                if top_doc["id"] not in {d["id"] for d in active_batch_history[i]}:
                    active_batch_history[i].append(top_doc)

        if log_trace:
            for question, query, history, docs in zip(active_questions, remaining_queries, active_batch_history, batch_docs):
                print(f"Iteration {iter_count+1} -- Question: {question['question']}")
                print("History:")
                for doc in history:
                    print(f"  Passage: {doc['text']}")
                print(f"Generated query: {query}")
                print("Retrieved passages:")
                for doc in docs:
                    print(f"  Passage: {doc['text']}")
                print("\n")

        print("Iteration", iter_count + 1, "completed in", time.time() - start_time, "seconds")
        print(f"Remaining active questions: {len(active_questions)}\n")

        iter_count += 1
        if iter_count >= max_iterations:
            final_questions.extend(active_questions)
            final_batch_history.extend(active_batch_history)
            break

    em_list = [[], [], []]
    precision_list = [[], [], []]
    recall_list = [[], [], []]
    f1_list = [[], [], []]

    for question, history in zip(final_questions, final_batch_history):
        question_id = question["id"]
        correct = sum(int(question_id in doc["id"]) for doc in history)
        num_hops = int(question_id[0])  
        num_retrieval = len(history)
        em_list[num_hops - 2].append(int(correct == num_hops and num_hops == num_retrieval))
        precision_list[num_hops - 2].append(correct / num_retrieval)
        recall_list[num_hops - 2].append(correct / num_hops)
        f1_list[num_hops - 2].append(2 * correct / (num_hops + num_retrieval))

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

    main_group = parser.add_argument_group("Main Options")
    main_group.add_argument("--questions", type=str, required=True, help="Questions file path")
    main_group.add_argument("--batch-size", type=int, default=32, help="Batch size for processing questions")
    main_group.add_argument("--max-iterations", type=int, default=5, help="Maximum number of iterations")
    main_group.add_argument("--max-search", type=int, default=10, help="Maximum number of passages to retrieve")
    main_group.add_argument("--log-trace", action="store_true", help="Log trace for debugging")
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

    with open(args.questions, "r") as f:
        questions = f.readlines()
        questions = [json.loads(q) for q in questions]

    em_list = [[], [], []]
    precision_list = [[], [], []]
    recall_list = [[], [], []]
    f1_list = [[], [], []]

    for i in range(0, len(questions), args.batch_size):
        batch_questions = questions[i : i + args.batch_size]
        print(f"Processing batch {i // args.batch_size + 1} of {len(questions) // args.batch_size + 1}...\n")

        em, precision, recall, f1 = run_batch(
            retriever=retriever,
            query_generator=query_generator,
            questions=batch_questions,
            max_iterations=args.max_iterations,
            max_search=args.max_search,
            log_trace=args.log_trace,
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
