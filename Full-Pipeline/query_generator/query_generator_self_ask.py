import os
import torch
from vllm import LLM, SamplingParams
from .prompts import SELF_ASK_PROMPT


class QueryGenerator:
    def __init__(self, model_id, tp_size, quantization, max_gen_length=200, temperature=0.7, top_p=0.9):
        os.environ['MKL_THREADING_LAYER']='GNU'

        self.llm = LLM(
            model=model_id,
            tensor_parallel_size=tp_size,
            quantization=quantization,
            gpu_memory_utilization=0.9,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_gen_length,
        )

        print("Model loaded successfully.")

    def _gen_retriever_query_prompt(self, trace, is_first=False):
        prompt = SELF_ASK_PROMPT + "\n" + trace
        if is_first:
            prompt += "Give a follow up question.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        else:
            prompt += (
                "Give an intermediate answer and a follow up question if needed. If not, give the final answer."
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            )
        return prompt.strip()

    def batch_generate(self, traces, is_first=False):
        prompts = [self._gen_retriever_query_prompt(trace, is_first) for trace in traces]

        outputs = self.llm.generate(prompts, self.sampling_params)

        new_traces = []
        queries = []
        for trace, output in zip(traces, outputs):
            new_trace, query = self.extract_query(output.outputs[0].text)
            trace += new_trace
            if query:
                new_traces.append(trace)
                queries.append(query)
            else:
                new_traces.append(trace)
                queries.append("")

        return new_traces, queries

    def extract_query(self, text):
        lines = text.strip().split('\n')
        trace = ""
        for line in lines:
            trace += line.strip() + "\n"
            if line.strip().lower().startswith("follow up: ") or line.strip().lower().startswith("follow-up: "):
                return trace, line.strip()[len("Follow up: "):]
        return trace, None


def test():
    query_generator = QueryGenerator(
        model_id="casperhansen/llama-3.3-70b-instruct-awq",
        tp_size=2,
        quantization="awq_marlin",
        max_gen_length=2048,
        temperature=0.7,
        top_p=0.9,
    )

    traces = [(
        "Question: Country A has an embassy from the country that produces the show Krystala. Who was a prominent figure in the radio division of the network that created a version of the Biggest Loser for country A?\n"
        "Are follow up questions needed here: Yes.\n"
        "Follow up: Which country produces the show Krystala?\n"
        "Context: Krystala is a daily fantasy/sci-fi/adventure/soap opera serial (superserye/fantaserye) from the Philippines, where it was produced by and aired on ABS-CBN from October 11, 2004 to April 22, 2005. The show also aired simultaneously on The Filipino Channel and on a one-week delay on International Channel (now AZN-TV) in the United States.\n"
    )]
    is_first = False

    new_traces, queries = query_generator.batch_generate(traces, is_first)
    for trace, query in zip(new_traces, queries):
        print("Trace: ")
        print(trace)
        print(f"Query: {query}")
        print("-" * 50)
    print("Test completed.")


if __name__ == "__main__":
    test()
