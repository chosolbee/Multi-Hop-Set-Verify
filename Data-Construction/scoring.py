import json

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

if __name__ == "__main__":
    input_path = "INPUT_FILE.jsonl"
    output_path = "OUTPUT_FILE.jsonl"
    lambda_val = 0.5
    process_data(input_path, output_path, lambda_val)