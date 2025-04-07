import json
import argparse

def process_data(input_path, output_path, alpha=0.3):
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            recall = data.get("coverage", 0)
            noise = data.get("noise", 0)
            precision = 1 - noise

            score = alpha * precision + (1 - alpha) * recall
            data["score"] = score

            fout.write(json.dumps(data, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute score using affine combination of Precision and Recall.")
    parser.add_argument("--input_file", required=True, help="Path to input JSONL file")
    parser.add_argument("--output_file", required=True, help="Path to output JSONL file")
    parser.add_argument("--alpha", type=float, default=0.3, help="Alpha value for weighting Precision vs Recall (default: 0.3)")

    args = parser.parse_args()
    process_data(args.input_file, args.output_file, args.alpha)