import json
import argparse

def process_data(input_path, output_path, lambda_val=0.5):
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            coverage = data.get("coverage", 0)
            noise = data.get("noise", 0)
            # score = coverage - lambda_val * noise
            data["score"] = coverage - lambda_val * noise
            fout.write(json.dumps(data, ensure_ascii=False) + "\n")

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Scoring",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # Shows defaults in help
    )
    
    parser.add_argument("-i", "--input", required=True, help="Input File Path", type=str)
    parser.add_argument("-o", "--output", required=True, help="Output File Path", type=str)
    parser.add_argument("-l", "--lambda", default=0.5, help="Lambda Value", type=float)
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    input_path = args.input
    output_path = args.output
    lambda_val = args.lambda

    process_data(input_path, output_path, lambda_val)
