import os
import re
import torch
from vllm import LLM, SamplingParams
from .prompts import COT_PROMPT


class QueryGenerator:
    def __init__(self, llm, max_gen_length=200, temperature=0.7, top_p=0.9):
        os.environ['MKL_THREADING_LAYER']='GNU'

        self.llm = llm

        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_gen_length,
        )

    def _gen_retriever_query_prompt(self, question, confirmed_passages):
        prompt = COT_PROMPT + question + "\n"
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

        outputs = self.llm.generate(prompts, self.sampling_params)

        clean_texts = [self.extract_query(output.outputs[0].text) for output in outputs]
        return clean_texts

    def extract_query(self, generated_text):
        pattern = r"<query>(.*?)</query>"
        match = re.search(pattern, generated_text, re.DOTALL)

        if match:
            return match.group(1).strip()
        else:
            return generated_text.strip()


def test():
    llm = LLM(
        model="meta-llama/Llama-3.1-8B-instruct",
        tensor_parallel_size=1,
        quantization=None,
        dtype=torch.bfloat16,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
    )

    query_generator = QueryGenerator(
        llm=llm,
        max_gen_length=2048,
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
    test()
