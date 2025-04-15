import re
import json
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer


class QueryGenerator:
    def __init__(self, model_id, cache_dir=None, max_gen_length=200, temperature=0.7, top_p=0.9):
        self.max_gen_length = max_gen_length
        self.temperature = temperature
        self.top_p = top_p

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            cache_dir=cache_dir, 
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )

    def _gen_retriever_query_prompt(self, question, confirmed_passages):
        prompt = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            "You are tasked with generating targeted retrieval queries for multi-hop questions. A multi-hop question requires multiple pieces of information to be answered fully.\n\n"
            "Given:\n"
            "- A multi-hop question\n"
            "- A set of confirmed, relevant passages (which may be empty)\n\n"
            "Your task:\n"
            "Generate a single retrieval query that targets ONE specific missing or complementary piece of evidence needed to answer the original question that is not covered by the provided passages.\n\n"
            "Guidelines:\n"
            "1. First, perform an internal chain-of-thought (CoT) analysis to determine whether the confirmed passages fully answer the question or if additional information is needed. (Do not include your CoT in the final output.)\n"
            "2. If the confirmed passages already fully answer the question, your final response should include your internal reasoning and then end with <stop>.\n"
            "3. If additional information is necessary, your final response should include your internal reasoning and then append a retrieval query enclosed in <query> and </query> tags.\n"
            "4. Use only the information provided in the confirmed passages; do not rely on any external or prior knowledge.\n\n"
            "[Information Sufficiency Checklist]\n"
            "- Do the confirmed passages contain all critical details (names, dates, events, locations, etc.) required to answer the question?\n"
            "- If all required details are present, your final response should end with <stop>.\n\n"
            "Few-Shot Examples:\n\n"
            "Example 1:\n"
            "Question: What is the capital of France?\n"
            "Your Response: The question asks for the capital of France, and the confirmed passage clearly states that the capital is Paris. All required details are provided. <stop>\n\n"
            "Example 2:\n"
            "Question: What's the meaning of the name of the school that does not include the Mahayava scriptures in its canon?\n"
            "Your Response: The question asks for the meaning of the school's name, but the confirmed passage does not provide the actual name of the school. I need the name to determine its meaning. <query>What is the name of the school that does not include the Mahayava scriptures in its canon?</query>\n\n"
            "Example 3:\n"
            "Question: Who was picked in the NBA draft before the player with the highest point average among active players?\n"
            "Confirmed Passage 1: \"Wilt Chamberlain holds the all-time records for total points scored (4,029) and points per game (50.4) in a season; he also holds the rookie record when he averaged 37.6 points in the 1959–60 season. Among active players, Kevin Durant has the highest point average (32.0) in a season, achieved in the 2013–14 season.\"\n"
            "Your Response: The question seeks to know who was picked in the NBA draft before the player with the highest point average among active players. The confirmed passage shows that Kevin Durant is that player but does not specify who was drafted before him. <query>Who was picked in the NBA draft before Kevin Durant?</query>\n\n"
            "Example 4:\n"
            "Question: When did the city that WIZE is licensed to broadcast to become the capital of the state that contains the county where the Battle of Rich Mountain occurred?\n"
            "Confirmed Passage 1: \"The Battle of Rich Mountain took place on July 11, 1861, in Randolph County, Virginia (now West Virginia) during the American Civil War.\"\n"
            "Confirmed Passage 2: \"Kaskaskia, located in that county, was Illinois's first capital.\"\n"
            "Confirmed Passage 3: \"WIZE (1340 AM) is a radio station based in Springfield, Ohio.\"\n"
            "Your Response: The confirmed passages provide historical context and indicate that Kaskaskia was Illinois's first capital, but they do not specify when Springfield became the capital of Illinois. <query>When did Springfield become the capital of Illinois?</query>\n\n"
            "Example 5 (2-Hop with Two Confirmed Passages – <stop> Output):\n"
            "Question: What river flows through Paris, and which iconic monument on its banks was completed in 1889?\n"
            "Confirmed Passage 1: \"The Seine River flows through the heart of Paris.\"\n"
            "Confirmed Passage 2: \"The Eiffel Tower, an iconic monument along the Seine, was completed in 1889.\"\n"
            "Your Response: The question asks for the river flowing through Paris and the iconic monument completed in 1889 on its banks. The confirmed passages provide both details: the Seine River and the Eiffel Tower. All required information is present. <stop>\n\n"
            "<|eot_id|>\n"
            "<|start_header_id|>user<|end_header_id|>\n"
            f"Question: {question}\n"
        )
        if confirmed_passages:
            for idx, passage in enumerate(confirmed_passages, start=1):
                prompt += f"Confirmed Passage {idx}: {passage['text']}\n"
        prompt += "<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
        return prompt.strip()



    def batch_generate(self, questions, batch_confirmed_passages):
        prompts = [
            self._gen_retriever_query_prompt(question["question"], confirmed_passages)
            for question, confirmed_passages in zip(questions, batch_confirmed_passages)
        ]
        inputs = self.tokenizer(prompts, padding=True, padding_side="left", return_tensors="pt").to(self.model.device)

        input_lengths = [i.size(0) for i in inputs["input_ids"]]
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_gen_length,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        generated_texts = self.tokenizer.batch_decode(
            [output[input_lengths[i]:] for i, output in enumerate(outputs)],
            skip_special_tokens=True,
        )
        clean_texts = [self.extract_query(text) for text in generated_texts]
        return clean_texts

    def extract_query(self, generated_text):
        if re.fullmatch(r"\s*<stop>\s*", generated_text, re.IGNORECASE):
            return "<stop>"

        pattern = r"<query>(.*?)</query>"
        match = re.search(pattern, generated_text, re.DOTALL)

        if match:
            return match.group(1).strip()
        else:
            return generated_text.strip()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cache-dir", type=str, help="Cache directory for model")
    args = parser.parse_args()
    return args


def test(args: argparse.Namespace):
    query_generator = QueryGenerator(
        model_id="meta-llama/Meta-Llama-3-8B-Instruct",
        cache_dir=args.cache_dir,
        max_gen_length=200,
        temperature=0.7,
        top_p=0.9,
    )
    questions = [{"question": "What is the capital of France?"}, {"question": "Explain the theory of relativity."}]
    batch_confirmed_passages = [
        [
            {"text": "The capital of France is Paris."},
            {"text": "The theory of relativity was developed by Albert Einstein."},
        ],
        [
            {"text": "The capital of France is Paris."},
            {"text": "The theory of relativity was developed by Albert Einstein."},
        ],
    ]

    generated_queries = query_generator.batch_generate(questions, batch_confirmed_passages)
    for question, query in zip(questions, generated_queries):
        print(f"Question: {question['question']}\nGenerated Query: {query}\n")


if __name__ == "__main__":
    args = parse_args()
    test(args)
