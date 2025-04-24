import random
import argparse

def split_data(input_path, eval_output_path, test_output_path, split_ratio=0.8, seed=42):
    with open(input_path, "r") as f:
        lines = f.readlines()

    random.seed(seed)
    random.shuffle(lines)

    split_idx = int(len(lines) * split_ratio)
    eval_lines = lines[:split_idx]
    test_lines = lines[split_idx:]

    with open(eval_output_path, "w") as f:
        f.writelines(eval_lines)

    with open(test_output_path, "w") as f:
        f.writelines(test_lines)

    print(f"Eval data written to {eval_output_path}")
    print(f"Test data written to {test_output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Split JSONL dataset into eval and test sets.")
    parser.add_argument("--input-path", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--eval-output-path", type=str, required=True, help="Path to save the eval set.")
    parser.add_argument("--test-output-path", type=str, required=True, help="Path to save the test set.")
    parser.add_argument("--split-ratio", type=float, default=0.5, help="Proportion of data to use for eval (default: 0.5).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling (default: 42).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    split_data(
        input_path=args.input_path,
        eval_output_path=args.eval_output_path,
        test_output_path=args.test_output_path,
        split_ratio=args.split_ratio,
        seed=args.seed,
    )