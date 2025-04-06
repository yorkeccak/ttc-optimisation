# fine_tuned_llm.py
# Fine-tuned LLM is a more sophisticated version of rag_token_llm.py that uses a fine-tuned model to output search tokens instead of prompt engineering.

from ..llm import Llm
from typing import Self, Generator
import torch
import threading
import time
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)
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

Today's date: April 5th, 2025. Be direct and concise.

### Response:
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

        self._ft_tokenizer = AutoTokenizer.from_pretrained(
            "canertugrul/DeepSeek-R1-Distill-Qwen-14B-Tool-Use-Tokenizer_v3",
            cache_dir="./.cache/huggingface/hub",
        )

        # Add these state tracking variables
        self._in_thinking = False
        self._thinking_start = None

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

        base_model.resize_token_embeddings(len(self._ft_tokenizer))

        # =============================
        # = ADD ADAPTER TO BASE MODEL =
        # =============================

        self._ft_model = PeftModel.from_pretrained(
            base_model,
            "canertugrul/DeepSeek-R1-Distill-Qwen-14B-Tool-Use-Adapter_v3",
            cache_dir="./.cache/huggingface/hub",
            device_map="auto",
        )

        self._ft_model.eval()

    def _run_inference_stream(
        self: Self, prompt: str, stop_tokens=None
    ) -> Generator[str, None, None]:
        # Prepare model for inference (no need for FastLanguageModel)
        self._ft_model.eval()

        inputs = self._ft_tokenizer([prompt], return_tensors="pt").to(
            self._ft_model.device
        )
        stop_token_id = self._ft_tokenizer.convert_tokens_to_ids(self._end_rag)
        stopping_criteria = StoppingCriteriaList([StopOnTokens([stop_token_id])])
        streamer = TextIteratorStreamer(
            self._ft_tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        generation_kwargs = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "max_new_tokens": 8192,
            "use_cache": True,
            "stopping_criteria": stopping_criteria,
            "pad_token_id": self._ft_tokenizer.eos_token_id,
            "streamer": streamer,
        }

        # Start generation in a separate thread
        thread = threading.Thread(
            target=self._ft_model.generate, kwargs=generation_kwargs
        )
        thread.start()

        # Track thinking state
        buffer = ""
        # Yield from the streamer
        for text in streamer:
            print(f"{text}", end="", flush=True)
            buffer += text

            # Check for thinking start and end in the buffer
            if not self._in_thinking and "<think>" in buffer:
                self._in_thinking = True
                self._thinking_start = time.time()
                print(f"\r🧠 Thinking started...", end="")

            # Get thinking time if incomplete thinking
            if self._in_thinking and (
                (self._end_rag in buffer) or ("</think>" in buffer)
            ):
                # Capture thinking time up to this point and restart for next turn
                current_thinking_time = time.time() - self._thinking_start
                self._thinking_times.append(current_thinking_time)
                if "</think>" in buffer:
                    self._in_thinking = False
                    self._thinking_start = None
                print(
                    f"\r✅ Thinking ended ({current_thinking_time:.2f} seconds)", end=""
                )
            yield text

    def generate_output(self: Self, question: str, max_turns: int = 5) -> dict:
        prompt = PROMPT_TEMPLATE.format(
            question=question,
            start_rag=self._start_rag,
            end_rag=self._end_rag,
            start_result=self._start_result,
            end_result=self._end_result,
        )
        print(f"Prompt: {prompt}")
        output = ""

        for turn in range(max_turns):
            print(f"\n--- Turn {turn+1} ---")
            # Stream the response
            response = ""
            print("Model Response: \n", end="")
            for chunk in self._run_inference_stream(prompt):
                response += chunk

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
            # Reset thinking for next turn after API call
            self._thinking_start = time.time()

        # Final cleanup in case thinking state is still active at the end
        self._in_thinking = False
        self._thinking_start = None

        return self._compute_metrics(output)
