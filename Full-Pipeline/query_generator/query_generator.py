import re
import json
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

class QueryGenerator():
    def __init__(self, model_id, cache_dir=None, max_gen_length=200, temperature=0.7, top_p=0.9):
        self.max_gen_length = max_gen_length
        self.temperature = temperature
        self.top_p = top_p

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            cache_dir=cache_dir, 
            trust_remote_code=True
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
            "- A set of already confirmed relevant passages (which may be empty)\n\n"
            "Your task:\n"
            "Generate a single retrieval query that targets ONE specific piece of missing or complementary evidence needed to answer the original question but not covered in the provided passages.\n\n"
            "Guidelines:\n"
            "- Create a single-hop question (answerable by referring to a single piece of information)\n"
            "- Be specific enough to retrieve relevant information but not overly restrictive\n"
            "- Maintain important entities and keywords from the original question\n"
            "- Focus on the most critical missing information\n"
            "- If multiple pieces of information are missing, prioritize the most fundamental one\n"
            "- DO NOT use any prior knowledge beyond the provided passages.\n\n"
            "You may add some additional explanation or commentary in your response, but make sure to wrap the final retrieval query between <query> and </query> tags.\n\n"
            "Here are some examples for your reference:\n\n"
            "Example 1:\n"
            "Question: What's the meaning of the name of the school that does not include the Mahayava scriptures in its canon?\n"
            "Your Response: The given question is asking about the meaning of the name of a specific school. To answer this, I need to know the name of the school that does not include the Mahayava scriptures in its canon. Therefore, my retrieval query would be: <query>What is the name of the school that does not include the Mahayava scriptures in its canon?</query>\n\n"            
            "Example 2:\n"
            "Question: Who founded the company that distributed the film UHF?\n"
            "Your Response: The given question is asking about the founder of a specific company. To answer this, I need to know the name of the company that distributed the film UHF. Therefore, my retrieval query would be: <query>What is the name of the company that distributed the film UHF?</query>\n\n"
            "Example 3:\n"
            "Question: Who was picked in the NBA draft before the player with the highest point average among active players?\n"
            "Confirmed Passage 1: Wilt Chamberlain holds the all - time records for total points scored (4,029) and points per game (50.4) in a season; both records were achieved in the 1961 -- 62 season. He also holds the rookie records for points per game when he averaged 37.6 points in the 1959 -- 60 season. Among active players, Kevin Durant has the highest point total (2,593) and the highest scoring average (32.0) in a season; both were achieved in the 2013 -- 14 season.\n"
            "Your Response: From the given passage, I can see that Kevin Durant has the highest point average among active players. To answer the question, I need to know who was picked in the NBA draft before him. Therefore, my retrieval query would be: <query>Who was picked in the NBA draft before Kevin Durant?</query>\n\n"
            "Example 4:\n"
            "Question: When did the city that WIZE is licensed to broadcast to, become the capital of the state, that contains the county where the Battle of Rich Mountain occurred?\n"
            "Confirmed Passage 1: The Battle of Rich Mountain took place on July 11, 1861, in Randolph County, Virginia (now West Virginia) as part of the Operations in Western Virginia Campaign during the American Civil War.\n"
            "Confirmed Passage 2: Owing to its role in the state's history, the county motto is \"Where Illinois Began.\" It contains the historically important village of Kaskaskia, Illinois's first capital.\n"
            "Confirmed Passage 3: WIZE (1340 AM) — branded WIZE AM 1340 — is a commercial radio station in Springfield, Ohio owned by iHeartMedia, Inc. as part of their Dayton cluster. The station's main format is classic country targeted towards Springfield, and their transmitter - and former studios - are also located in Springfield.\n"
            "Your Response: From the given passages, I can see that the Battle of Rich Mountain occurred in Randolph County, Virginia (now West Virginia), and that Kaskaskia was Illinois's first capital. To answer the question, I need to know when Springfield became the capital of Illinois. Therefore, my retrieval query would be: <query>When did Springfield become the capital of Illinois?</query>\n"
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
        prompts = [self._gen_retriever_query_prompt(question["question"], confirmed_passages) for question, confirmed_passages in zip(questions, batch_confirmed_passages)]
        inputs = self.tokenizer(prompts, padding=True, padding_side="left", return_tensors="pt").to(self.model.device)

        input_lengths = [i.size(0) for i in inputs["input_ids"]]
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_gen_length,
            temperature=self.temperature,
            top_p=self.top_p
        )
        generated_texts = self.tokenizer.batch_decode(
            [output[input_lengths[i]:] for i, output in enumerate(outputs)],
            skip_special_tokens=True
        )
        clean_texts = [self.extract_query(text) for text in generated_texts]
        return clean_texts

    def extract_query(self, generated_text):
        pattern = r"<query>(.*?)</query>"
        match = re.search(pattern, generated_text, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        else:
            return generated_text.strip()


def test():
    query_generator = QueryGenerator(
        model_id="meta-llama/Meta-Llama-3-8B-Instruct",
        max_gen_length=200,
        temperature=0.7,
        top_p=0.9
    )
    questions = [{"question": "What is the capital of France?"}, {"question": "Explain the theory of relativity."}]
    batch_confirmed_passages = [
        [
            {"text": "The capital of France is Paris."},
            {"text": "The theory of relativity was developed by Albert Einstein."}
        ],
        [
            {"text": "The capital of France is Paris."},
            {"text": "The theory of relativity was developed by Albert Einstein."}
        ]
    ]

    generated_queries = query_generator.batch_generate(questions, batch_confirmed_passages)
    for question, query in zip(questions, generated_queries):
        print(f"Question: {question['question']}\nGenerated Query: {query}\n")


if __name__ == "__main__":
    test()
