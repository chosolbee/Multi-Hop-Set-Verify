import json
import argparse
import wandb
from config import WANDB_ENTITY, DEBERTA_MAX_LENGTH
from .contriever import Retriever
from .query_generator import QueryGenerator
from .verifier import Verifier


def run_batch(retriever, query_generator, verifier, questions, max_iterations=5, max_search=10, verifier_threshold=9.0):
    final_questions = []
    final_batch_history = []

    batch_history = [[] for _ in range(len(questions))]

    iter_count = 0
    while questions:
        queries = query_generator.batch_generate(questions, batch_history)
        batch_docs = retriever.search(queries, max_search)
        batch_scores = verifier.batch_verify(questions, batch_history, batch_docs)

        print("Iteration", iter_count + 1, "\n")
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
            max_score = scores.max()
            max_idx = scores.argmax()
            history.append(docs[max_idx])

            if max_score > verifier_threshold:
                final_questions.append(question)
                final_batch_history.append(history)
            else:
                next_questions.append(question)
                next_batch_history.append(history)

        questions = next_questions
        batch_history = next_batch_history

        iter_count += 1
        if iter_count >= max_iterations:
            final_questions.extend(questions)
            final_batch_history.extend(batch_history)
            break

    print("\nFinal Questions and History:\n")
    for question, history in zip(final_questions, final_batch_history):
        print(f"1. Question: {question}")
        print("2. History:")
        for doc in history:
            print(f"  Passage: {doc['text']}")
        print("\n")


def parse_args():
    parser = argparse.ArgumentParser()

    retriever_group = parser.add_argument_group("Retriever Options")
    retriever_group.add_argument("--passages", type=str, required=True, help="document file path")
    retriever_group.add_argument("--embeddings", type=str, required=True, help="Document embedding path")

    query_generator_group = parser.add_argument_group("Query Generator Options")
    query_generator_group.add_argument("--qg-cache-dir", type=str, help="Cache directory for query generator model")
    query_generator_group.add_argument("--qg-max-gen-length", type=int, default=200, help="Maximum generation length for query generator")
    query_generator_group.add_argument("--qg-temperature", type=float, default=0.7, help="Temperature for query generator")
    query_generator_group.add_argument("--qg-top-p", type=float, default=0.9, help="Top-p sampling for query generator")

    verifier_group = parser.add_argument_group("Verifier Options")
    verifier_group.add_argument("--verifier-checkpoint-path", type=str, required=True, help="Checkpoint path for trained model")
    verifier_group.add_argument("--verifier-batch-size", type=int, default=8, help="Batch size for verifier")
    verifier_group.add_argument("--verifier-max-length", type=int, default=DEBERTA_MAX_LENGTH, help="Maximum length for verifier input")

    main_group = parser.add_argument_group("Main Options")
    main_group.add_argument("--questions", type=str, required=True, help="Questions file path")
    main_group.add_argument("--max-iterations", type=int, default=5, help="Maximum number of iterations")
    main_group.add_argument("--max-search", type=int, default=10, help="Maximum number of passages to retrieve")
    main_group.add_argument("--verifier-threshold", type=float, default=9.0, help="Threshold for verifier scores")
    args = parser.parse_args()
    return args


def main(args: argparse.Namespace):
    wandb.init(project="MultiHopQA-test", entity=WANDB_ENTITY)

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

    with open(opt.questions, "r") as f:
        questions = f.readlines()
        questions = [json.loads(q) for q in questions]

    run_batch(retriever, query_generator, verifier, questions)

    print("All done!")

    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)
