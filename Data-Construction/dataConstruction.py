import json
import random
import itertools
import argparse

def compute_metrics(candidate_ids, gold_ids):
    if not candidate_ids:
        return 0.0, 0.0
    num_gold_in_candidate = len(set(candidate_ids) & set(gold_ids))
    coverage = num_gold_in_candidate / len(gold_ids) if gold_ids else 0.0
    noise = (len(candidate_ids) - num_gold_in_candidate) / len(candidate_ids)
    return coverage, noise

def sample_incomplete_sets(gold_ids, max_incomplete=5):
    incomplete_sets = []
    k = len(gold_ids)
    if k <= 1:
        return incomplete_sets
    for r in range(1, k):
        for subset in itertools.combinations(gold_ids, r):
            incomplete_sets.append(set(subset))
    if len(incomplete_sets) > max_incomplete:
        incomplete_sets = random.sample(incomplete_sets, max_incomplete)
    return incomplete_sets

def sample_noise_sets(all_ids, gold_ids, max_noise=5, candidate_size_choices=None):
    noise_sets = set()
    max_tries = 1000
    tries = 0
    while len(noise_sets) < max_noise and tries < max_tries:
        tries += 1
        size = random.choice(candidate_size_choices)
        candidate = set(random.sample(all_ids, size))
        if candidate.issubset(set(gold_ids)):
            continue
        noise_sets.add(frozenset(candidate))
    return [set(s) for s in noise_sets]

def construct_candidate_instances(instance, max_incomplete=5, max_noise=5):
    if "gold_passages" in instance:
        gold_ids = instance["gold_passages"]
    else:
        gold_ids = [p["idx"] for p in instance.get("paragraphs", []) if p.get("is_supporting", False)]
        instance["gold_passages"] = gold_ids
    all_ids = [p["idx"] for p in instance.get("paragraphs", [])]
    
    candidate_instances = []
    hop = len(gold_ids)
    question_id = instance.get("id", None)
    
    complete_set = set(gold_ids)
    coverage, noise = compute_metrics(complete_set, gold_ids)
    candidate_instances.append({
         "question_id": question_id,
         "question": instance["question"],
         "gold_passages": gold_ids,
         "hop": hop,
         "passages": [p for p in instance["paragraphs"] if p["idx"] in complete_set],
         "set_type": "complete",
         "coverage": coverage,
         "noise": noise
    })
    
    incomplete_candidate_sets = sample_incomplete_sets(gold_ids, max_incomplete=max_incomplete)
    for cand in incomplete_candidate_sets:
         coverage, noise = compute_metrics(cand, gold_ids)
         candidate_instances.append({
             "question_id": question_id,
             "question": instance["question"],
             "gold_passages": gold_ids,
             "hop": hop,
             "passages": [p for p in instance["paragraphs"] if p["idx"] in cand],
             "set_type": "incomplete",
             "coverage": coverage,
             "noise": noise
         })
    
    candidate_size_choices = list(range(1, len(gold_ids) + 2))
    noise_candidate_sets = sample_noise_sets(all_ids, gold_ids, max_noise=max_noise, candidate_size_choices=candidate_size_choices)
    for cand in noise_candidate_sets:
         coverage, noise = compute_metrics(cand, gold_ids)
         candidate_instances.append({
             "question_id": question_id,
             "question": instance["question"],
             "gold_passages": gold_ids,
             "hop": hop,
             "passages": [p for p in instance["paragraphs"] if p["idx"] in cand],
             "set_type": "noise",
             "coverage": coverage,
             "noise": noise
         })
    return candidate_instances

def print_candidate_instance_summary(candidate):
    passage_ids = [p["idx"] for p in candidate.get("passages", [])]
    print(f"Question ID: {candidate.get('question_id', 'N/A')}")
    print(f"Question: {candidate['question']}")
    print(f"Set Type: {candidate['set_type']}")
    print(f"Hop: {candidate['hop']}")
    print(f"Passage IDs: {passage_ids}")
    print(f"Coverage: {candidate['coverage']}, Noise: {candidate['noise']}")
    print("-" * 50)

def process_file(input_path, output_path, max_incomplete=5, max_noise=5):
    new_data = []
    with open(input_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            if not line.strip():
                continue
            instance = json.loads(line)
            candidate_instances = construct_candidate_instances(instance, max_incomplete, max_noise)
            new_data.extend(candidate_instances)
            for candidate in candidate_instances:
                print_candidate_instance_summary(candidate)
    with open(output_path, 'w', encoding='utf-8') as fout:
        for candidate in new_data:
            fout.write(json.dumps(candidate, ensure_ascii=False) + "\n")
    print(f"Processed {len(new_data)} candidate instances. Saved to {output_path}")

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Scoring",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # Shows defaults in help
    )
    
    parser.add_argument("-i", "--input", required=True, help="Input File Path", type=str)
    parser.add_argument("-o", "--output", required=True, help="Output File Path", type=str)
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    input_path = args.input
    output_path = args.output

    process_file(input_path, output_path)
