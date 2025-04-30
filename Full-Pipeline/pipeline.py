import os
import re
import time
import json
import argparse
import wandb
from collections import Counter
from vllm import LLM, SamplingParams
import torch

from config import WANDB_ENTITY, DEBERTA_MAX_LENGTH
from .contriever import Retriever
from .query_generator import QueryGenerator
from .verifier import Verifier
from .answer_generator import AnswerGenerator


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


def print_answer_metrics(em_all, f1_all):
    if not em_all:
        return
    print(f"Answer EM : {sum(em_all) / len(em_all):.4f}")
    print(f"Answer F1 : {sum(f1_all) / len(f1_all):.4f}\n")


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

    if log_trace:
        print("\nFinal Questions and History:\n")
        for question, history in zip(final_questions, final_batch_history):
            print(f"1. Question: {question['question']}")
            print("2. History:")
            for doc in history:
                print(f"  Passage: {doc['text']}")
            print()

    em_list = [[], [], []]
    precision_list = [[], [], []]
    recall_list = [[], [], []]
    f1_list = [[], [], []]

    for question, history in zip(final_questions, final_batch_history):
        qid = question["id"]
        gold_hop = len(question.get("question_decomposition", []))
        correct = sum(int(qid + "-sf" in doc["id"]) for doc in history)
        retrieved = len(history)

        em = int(correct == gold_hop and retrieved == gold_hop)
        precision = correct / retrieved if retrieved else 0.0
        recall = correct / gold_hop if gold_hop else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        # # DEBUG
        # print(f"[DEBUG] qid={qid}, gold_hop={gold_hop}, history_ids={[d['id'] for d in history]}, "
        #       f"correct={correct}, retrieved={retrieved}, em={em:.3f}, recall={recall:.3f}, precision={precision:.3f}")

        for log in stop_logs:
            if log["question_id"] == qid:
                log.update({"em": em, "precision": precision, "recall": recall, "f1": f1})
                break

        idx = max(gold_hop - 2, 0)
        em_list[idx].append(em)
        precision_list[idx].append(precision)
        recall_list[idx].append(recall)
        f1_list[idx].append(f1)

    if stop_log_path:
        with open(stop_log_path,'a',encoding='utf-8') as f:
            for log in stop_logs:
                f.write(json.dumps(log,ensure_ascii=False)+'\n')

    return em_list, precision_list, recall_list, f1_list, final_questions, final_batch_history


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

    verifier_group = parser.add_argument_group("Verifier Options")
    verifier_group.add_argument("--verifier-model-id", type=str, default="microsoft/DeBERTa-v3-large", help="Model ID for verifier")
    verifier_group.add_argument("--verifier-checkpoint-path", type=str, required=True, help="Checkpoint path for trained model")
    verifier_group.add_argument("--verifier-batch-size", type=int, default=8, help="Batch size for verifier")
    verifier_group.add_argument("--verifier-max-length", type=int, default=DEBERTA_MAX_LENGTH, help="Maximum length for verifier input")

    answer_generator_group = parser.add_argument_group("Answer Generator Options")
    answer_generator_group.add_argument("--generate-answers", action="store_true")
    answer_generator_group.add_argument("--ag-max-gen-length",type=int, default=1024, help="Maximum generation length for answer generator")
    answer_generator_group.add_argument("--ag-temperature", type=float, default=0.7, help="Temperature for answer generator")
    answer_generator_group.add_argument("--ag-top-p", type=float, default=0.9, help="Top-p sampling for answer generator")

    main_group = parser.add_argument_group("Main Options")
    main_group.add_argument("--questions", type=str, required=True, help="Questions file path")
    main_group.add_argument("--batch-size", type=int, default=32, help="Batch size for processing questions")
    main_group.add_argument("--max-iterations", type=int, default=5, help="Maximum number of iterations")
    main_group.add_argument("--max-search", type=int, default=10, help="Maximum number of passages to retrieve")
    main_group.add_argument("--verifier-threshold", type=float, default=0.9, help="Threshold for verifier scores")
    main_group.add_argument("--log-trace", action="store_true", help="Log trace for debugging")
    main_group.add_argument("--stop-log-path", type=str, default=None, help="Optional JSONL path; Path to the JSONL file where stopping logs are written")

    args = parser.parse_args()
    return args


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
        open(args.stop_log_path,"w", encoding="utf-8").close()

    with open(args.questions, "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]

    em_list = [[], [], []]
    precision_list = [[], [], []]
    recall_list = [[], [], []]
    f1_list = [[], [], []]

    ans_em_list  = [[], [], []]
    ans_f1_list  = [[], [], []] 

    for i in range(0, len(questions), args.batch_size):
        batch_questions = questions[i: i + args.batch_size]
        print(f"Processing batch {i // args.batch_size + 1} of {len(questions) // args.batch_size + 1}...\n")

        em, precision, recall, f1, final_q, final_hist = run_batch(
            retriever=retriever,
            query_generator=query_generator,
            verifier=verifier,
            questions=batch_questions,
            max_iterations=args.max_iterations,
            max_search=args.max_search,
            verifier_threshold=args.verifier_threshold,
            log_trace=args.log_trace,
            stop_log_path=args.stop_log_path,
        )
        for j in range(3):
            em_list[j].extend(em[j])
            precision_list[j].extend(precision[j])
            recall_list[j].extend(recall[j])
            f1_list[j].extend(f1[j])

        if args.generate_answers: 
            preds = answer_generator.batch_answer(final_q, final_hist)

            if len(preds) != len(final_q):
                print(f"[WARNING] Number of predictions ({len(preds)}) != number of questions ({len(final_q)})")

            for q, pred in zip(final_q, preds):
                hop      = len(q.get("question_decomposition", []))
                bucket   = max(hop - 2, 0)  
                golds    = [q["answer"]] + q.get("answer_aliases", [])
                golds    = [normalize(g) for g in golds]

                pn       = normalize(pred)
                em_ok    = int(pn in golds)
                f1_best  = max(token_f1(pn, g) for g in golds)

                print(
                    f"[ANS-DEBUG] qid={q['id']} | gold={golds} | "
                    f"pred={pn} | EM={em_ok:.0f}, F1={f1_best:.3f}"
                )

                ans_em_list[bucket].append(em_ok)
                ans_f1_list[bucket].append(f1_best) 

        print("===== RETRIEVAL PERFORMANCE =====")
        print_results(em_list, precision_list, recall_list, f1_list)
        if args.generate_answers:
            dummy_prec_rec = [
            [0.0] if len(bucket) == 0 else [0.0] * len(bucket)
            for bucket in ans_em_list
        ]
        print("===== ANSWER  PERFORMANCE =====")
        print_results(ans_em_list, dummy_prec_rec, dummy_prec_rec, ans_f1_list)

    print("===== FINAL RETRIEVAL METRICS =====")
    print_results(em_list, precision_list, recall_list, f1_list)

    print("===== FINAL ANSWER  METRICS =====")
    if args.generate_answers:
        dummy_prec_rec = [
            [0.0] if len(bucket) == 0 else [0.0] * len(bucket)
            for bucket in ans_em_list
        ]
        print_results(ans_em_list, dummy_prec_rec, dummy_prec_rec, ans_f1_list)

    
    print("All done!")

    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)