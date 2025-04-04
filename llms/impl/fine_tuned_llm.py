# fine_tuned_llm.py
# Fine-tuned LLM is a more sophisticated version of rag_token_llm.py that uses a fine-tuned model to output search tokens instead of prompt engineering.

import time
from ..llm import Llm
from typing import Self
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel


PROMPT_TEMPLATE = """
QUESTION: {question}

Answer this question accurately. If you're uncertain about any facts:
1. IMMEDIATELY output a search query between {start_rag} and {end_rag} tags
2. Wait for search results between {start_result} and {end_result}
3. Use these results to complete your answer

EXAMPLES:
Q: "Who is the current CEO of OpenAI?"
{start_rag}current CEO of OpenAI 2025{end_rag}
{start_result}Sam Altman returned as OpenAI CEO in November 2023 and continues to serve in this role as of March 2025.{end_result}
The current CEO of OpenAI is Sam Altman.

Q: "What is the population of Tokyo?"
{start_rag}current population of Tokyo 2025{end_rag}
{start_result}Tokyo's population is approximately 13.96 million as of January 2025.{end_result}
Tokyo has a population of approximately 13.96 million people.

Today's date: April 1, 2025. Be direct and concise.
"""

class FineTunedLlm(Llm):
    def __init__(self, model: str) -> None:
        super().__init__("")
        self._rag_enabled = True
        # Fix tokenizer initialization - should happen before using it
        self.tokenizer = AutoTokenizer.from_pretrained(
            "canertugrul/DeepSeek-R1-Distill-Qwen-14B-Tool-Use-Tokenizer", 
            cache_dir="./.cache/huggingface/hub",
        )

        # ===================
        # = LOAD BASE MODEL =
        # ===================
        load_in_4bit = True

        base_model = AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            cache_dir="./.cache/huggingface/hub",
            load_in_4bit = True,
            device_map="auto",
        )

        base_model.resize_token_embeddings(len(self.tokenizer))
        # ============================
        # = ADD TOKENS TO BASE MODEL =
        # ============================

        # Define special tokens
        special_tokens = {
            self._start_rag,
            self._end_rag,
            self._start_result,
            self._end_result
        }

        # =============================
        # = ADD ADAPTER TO BASE MODEL =
        # =============================

        self.model = PeftModel.from_pretrained(base_model, "canertugrul/DeepSeek-R1-Distill-Qwen-14B-Tool-Use-Adapter", cache_dir="./.cache/huggingface/hub", device_map='auto')

        self.model.eval()

   
    def _run_ollama(self: Self, prompt: str, stop_tokens=None):
        class StopOnTokens(StoppingCriteria):
            def __init__(self, stop_ids):
                self.stop_ids = stop_ids

            def __call__(self, input_ids, scores, **kwargs):
                return any(input_ids[0, -1].item() == stop_id for stop_id in self.stop_ids)

        # Prepare model for inference (no need for FastLanguageModel)
        self.model.eval()
        
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        stop_token_id = self.tokenizer.convert_tokens_to_ids(self._end_rag)

        stopping_criteria = StoppingCriteriaList([
            StopOnTokens([stop_token_id])
        ])

        outputs = self.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=8192,
            use_cache=True,
            stopping_criteria=stopping_criteria
        )

        # skip the prompt tokens
        generated_tokens = outputs[0][inputs.input_ids.shape[-1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

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
        print("Printing prompt:")
        for turn in range(max_turns):
            print(f"\n--- Turn {turn+1} ---")
            response = self._run_ollama(prompt)
            print(f"Model Response:\n{response}")

            output += "\n" + response
            search_query = self._extract_rag_query(response)
            
            if not search_query:
                print("\n[✅ No further searches required.]")
                print("\nFinal Response:\n")
                print(response)
                break

            # print(f"\n🔍 Searching: '{search_query}'")
            # res = self._run_valyu(search_query)
            res = ""
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