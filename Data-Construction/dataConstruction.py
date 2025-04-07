import json
import random
import itertools
import argparse

def compute_metrics(candidate_ids, gold_ids):
    if not candidate_ids:
        return 0.0, 0.0
    intersection = len(set(candidate_ids) & set(gold_ids))
    coverage = intersection / len(gold_ids) if gold_ids else 0.0
    noise = (len(candidate_ids) - intersection) / len(candidate_ids)
    return coverage, noise

def sample_incomplete_sets(gold_ids, max_samples):
    k = len(gold_ids)
    if k <= 1:
        return []
    all_subsets = [set(s) for r in range(1, k) for s in itertools.combinations(gold_ids, r)]
    return random.sample(all_subsets, min(len(all_subsets), max_samples))

def sample_noise_only_set(all_ids, gold_ids):
    noise_pool = list(set(all_ids) - set(gold_ids))
    if not noise_pool:
        return set()
    size = random.choice([s for s in range(1, 6) if s <= len(noise_pool)])
    return set(random.sample(noise_pool, size))

def sample_noise_with_gold_sets(all_ids, gold_ids, max_samples):
    noise_pool = list(set(all_ids) - set(gold_ids))
    results = set()
    while len(results) < max_samples:
        if not gold_ids:
            break
        seed_size = random.choice([s for s in range(1, min(5, len(gold_ids)+1))])
        seed = set(random.sample(gold_ids, seed_size))
        noise_limit = min(5 - seed_size, 2)
        noise_count = random.choice([n for n in [1, 2] if n <= noise_limit and n <= len(noise_pool)])
        noise = set(random.sample(noise_pool, noise_count))
        results.add(frozenset(seed | noise))
    return [set(s) for s in results]

def build_candidate(question_id, question, gold_ids, hop, paragraphs, selected_ids, set_type):
    coverage, noise = compute_metrics(selected_ids, gold_ids)
    return {
        "question_id": question_id,
        "question": question,
        "gold_passages": gold_ids,
        "hop": hop,
        "passages": [p for p in paragraphs if p["idx"] in selected_ids],
        "set_type": set_type,
        "coverage": coverage,
        "noise": noise
    }

def construct_candidate_instances(instance, max_incomplete=5, max_noise_with_gold=5):
    paragraphs = instance.get("paragraphs", [])
    gold_ids = instance.get("gold_passages") or [p["idx"] for p in paragraphs if p.get("is_supporting")]
    instance["gold_passages"] = gold_ids
    all_ids = [p["idx"] for p in paragraphs]
    qid, question, hop = instance.get("id"), instance["question"], len(gold_ids)

    candidates = []

    # complete
    candidates.append(build_candidate(qid, question, gold_ids, hop, paragraphs, set(gold_ids), "complete"))

    # incomplete
    for cand in sample_incomplete_sets(gold_ids, max_incomplete):
        candidates.append(build_candidate(qid, question, gold_ids, hop, paragraphs, cand, "incomplete"))

    # noise_only
    noise_only = sample_noise_only_set(all_ids, gold_ids)
    if noise_only:
        candidates.append(build_candidate(qid, question, gold_ids, hop, paragraphs, noise_only, "noise_only"))

    # noise_with_gold
    for cand in sample_noise_with_gold_sets(all_ids, gold_ids, max_noise_with_gold):
        candidates.append(build_candidate(qid, question, gold_ids, hop, paragraphs, cand, "noise_with_gold"))

    return candidates

def print_candidate_instance_summary(candidate):
    ids = [p["idx"] for p in candidate.get("passages", [])]
    print(f"[{candidate['set_type'].upper()}] QID: {candidate['question_id']} | Hop: {candidate['hop']} | IDs: {ids} | Coverage: {candidate['coverage']:.2f}, Noise: {candidate['noise']:.2f}")

def process_file(input_path, output_path, max_incomplete=5, max_noise_with_gold=5):
    all_candidates = []
    with open(input_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            if not line.strip():
                continue
            instance = json.loads(line)
            candidates = construct_candidate_instances(instance, max_incomplete, max_noise_with_gold)
            all_candidates.extend(candidates)
            for c in candidates:
                print_candidate_instance_summary(c)
    with open(output_path, 'w', encoding='utf-8') as fout:
        for c in all_candidates:
            fout.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"Saved {len(all_candidates)} candidate instances to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--max_incomplete", type=int, default=5)
    parser.add_argument("--max_noise_with_gold", type=int, default=5)
    args = parser.parse_args()

    process_file(args.input_file, args.output_file, args.max_incomplete, args.max_noise_with_gold)
