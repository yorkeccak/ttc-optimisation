# fine_tuned_llm.py
# Fine-tuned LLM is a more sophisticated version of rag_token_llm.py that uses a fine-tuned model to output search tokens instead of prompt engineering.

from ..llm import Llm
from typing import Self
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)
from peft import PeftModel

PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context.

Write a response that appropriately completes the request.

Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

If the answer relies on obscure concepts, knowledge unlikely to be in pretraining, or time-sensitive facts, trigger retrieval by:
1.⁠ ⁠Output a search query between {start_rag} and {end_rag} tags
2.⁠ ⁠Wait for search results between {start_result} and {end_result}
3.⁠ ⁠Use these results to complete your answer

Multiple retrievals may be included in a single Chain of Thought, including multiple lookups for the same topic (e.g., clarification followed by fact checking).

Today's date is April 1, 2025.

### Instruction:
You are a scientific expert with advanced knowledge in analytical reasoning, problem-solving, and quantitative analysis across various disciplines, including mathematics, physics, biology, chemistry, and engineering. Please answer the following question.

### Question:
{question}

### Response:
"""


class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_ids):
        self.stop_ids = stop_ids

    def __call__(self, input_ids, scores, **kwargs):
        return any(input_ids[0, -1].item() == stop_id for stop_id in self.stop_ids)


class FineTunedLlm(Llm):
    def __init__(self: Self, model: str) -> None:
        super().__init__(model)
        self._start_rag = "<search_query>"
        self._end_rag = "</search_query>"
        self._start_result = "<search_result>"
        self._end_result = "</search_result>"

        self.tokenizer = AutoTokenizer.from_pretrained(
            "canertugrul/DeepSeek-R1-Distill-Qwen-14B-Tool-Use-Tokenizer_v3",
            cache_dir="./.cache/huggingface/hub",
        )

        # ===================
        # = LOAD BASE MODEL =
        # ===================

        base_model = AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            cache_dir="./.cache/huggingface/hub",
            load_in_4bit=True,
            device_map="auto",
            bnb_4bit_compute_dtype=torch.float16,
        )

        base_model.resize_token_embeddings(len(self.tokenizer))

        # =============================
        # = ADD ADAPTER TO BASE MODEL =
        # =============================

        self.model = PeftModel.from_pretrained(
            base_model,
            "canertugrul/DeepSeek-R1-Distill-Qwen-14B-Tool-Use-Adapter_v3",
            cache_dir="./.cache/huggingface/hub",
            device_map="auto",
        )

        self.model.eval()

    def _run_inference(self: Self, prompt: str, stop_tokens=None):
        # Prepare model for inference (no need for FastLanguageModel)
        self.model.eval()

        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        stop_token_id = self.tokenizer.convert_tokens_to_ids(self._end_rag)
        stopping_criteria = StoppingCriteriaList([StopOnTokens([stop_token_id])])

        outputs = self.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=8192,
            use_cache=True,
            stopping_criteria=stopping_criteria,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # skip the prompt tokens
        generated_tokens = outputs[0][inputs.input_ids.shape[-1] :]
        response = self.tokenizer.decode(generated_tokens)

        return response

    def generate_output(self: Self, question: str, max_turns: int = 5) -> dict:
        prompt = PROMPT_TEMPLATE.format(
            question=question,
            start_rag=self._start_rag,
            end_rag=self._end_rag,
            start_result=self._start_result,
            end_result=self._end_result,
        )

        output = ""

        for turn in range(max_turns):
            print(f"\n--- Turn {turn+1} ---")
            response = self._run_inference(prompt)
            print(f"Model Response:\n{response}")

            output += "\n" + response
            search_query = self._extract_rag_query(response)

            if not search_query:
                print("\n[✅ No further searches required.]")
                print("\nFinal Response:\n")
                print(response)
                break

            print(f"\n🔍 Searching: '{search_query}'")
            res = self._run_valyu(search_query)
            print("\n📚 Search Results:")
            print("-" * 50)
            print(res)
            print("-" * 50)

            focus_reminder = f"""
Use the search results above to help answer the original question: "{question}"
Do not just summarize or explain the search results - use them to provide a direct answer to the question.
If you need additional information, you can search again
"""

            embedded_context = (
                f"\n{self._start_result}\n{res}\n{self._end_result}\n{focus_reminder}"
            )
            prompt += f"\n{response}\n{embedded_context}\n"
            print(f"\n📎 Search results added to context. Continuing reasoning...\n")

        return self._compute_metrics(output)
