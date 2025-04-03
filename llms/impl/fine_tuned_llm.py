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
Below is an instruction that describes a task, paired with an input that provides further context.

Write a response that appropriately completes the request.

Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a scientific expert with advanced knowledge in analytical reasoning, problem-solving, and quantitative analysis across various disciplines, including mathematics, physics, biology, chemistry, and engineering. Please answer the following question.

Immediately output the search query in the format <SEARCH>search query</SEARCH> to find relevant information.

### Question:
{question}

### Response:

"""

class FineTunedLlm(Llm):
    def __init__(self, model: str) -> None:
        super().__init__("")
        self._rag_enabled = True
        self._start_rag = "<SEARCH>"
        self._end_rag = "</SEARCH>"
        self._start_result = "<RESULT>"
        self._end_result = "</RESULT>"

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
        # Exclude existing tokens
        new_tokens = list(special_tokens - set(self.tokenizer.vocab.keys()))
        if new_tokens:
            print("Addding new tokens... ")
            # Add new tokens to tokenizer and model using add_new_tokens()
            self.add_new_tokens(base_model, self.tokenizer, new_tokens=new_tokens)
            # Verify the additions
            print(f"Added {len(new_tokens)} new tokens to the tokenizer and model! ")
        else:
            print("No new tokens added. ")

        self.model.eval()

    def add_new_tokens(self, model, tokenizer, new_tokens=[], method="mean", interpolation=0.5):
        assert isinstance(new_tokens, (list, tuple))
        assert len(new_tokens) > 0
        assert method in ["mean", "interpolation"]
        assert 0 <= interpolation <= 1
        overlapping_tokens = set(new_tokens) & set(tokenizer.vocab.keys())
        
        if overlapping_tokens:
            print(f"Unsloth: Skipping overlapping tokens: {list(overlapping_tokens)}")
            new_tokens = [x for x in new_tokens if x not in overlapping_tokens]

        # Add new tokens to tokenizercache
        old_length = len(tokenizer)
        tokenizer.add_tokens(new_tokens)

        # Fix — resize before accessing embedding matrix
        model.resize_token_embeddings(len(tokenizer))

        # Get mean embedding
        embedding_matrix = model.get_input_embeddings().weight.clone()
        lm_head_matrix = model.get_output_embeddings().weight.clone()
        eps = 1e-16
        indicator_untrained = torch.amax(embedding_matrix, axis=1) <= eps
        where_untrained = torch.where(indicator_untrained)[0]
        n_untrained = where_untrained.shape[0]
        n_trained = embedding_matrix.shape[0] - n_untrained
        sum_embedding = embedding_matrix.sum(dim=0) - embedding_matrix[where_untrained].sum(dim=0)
        sum_lm_head = lm_head_matrix.sum(dim=0) - lm_head_matrix[where_untrained].sum(dim=0)
        mean_embedding = (sum_embedding / n_trained).to(torch.float32)
        mean_lm_head = (sum_lm_head / n_trained).to(torch.float32)
        embedding_matrix = model.get_input_embeddings().weight
        lm_head_matrix = model.get_output_embeddings().weight

        if method == "interpolation":
            print("Using interpolation for initialising new tokens.")
            for j, token in enumerate(new_tokens):
                input_ids = tokenizer(token, add_special_tokens=False).input_ids
                token_mean_emb = embedding_matrix[input_ids].mean(dim=0)
                token_mean_head = lm_head_matrix[input_ids].mean(dim=0)
                emb = mean_embedding * (1 - interpolation) + token_mean_emb * interpolation
                head = mean_lm_head * (1 - interpolation) + token_mean_head * interpolation
                embedding_matrix[old_length + j] = emb
                lm_head_matrix[old_length + j] = head
        else:
            embedding_matrix.data[old_length:] = mean_embedding
            lm_head_matrix.data[old_length:] = mean_lm_head

        model.config.vocab_size = len(tokenizer)
        
        if hasattr(model, "tie_weights"):
            model.tie_weights()

        for _ in range(3):
            gc.collect()
            torch.cuda.empty_cache()
    
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

        response = self.tokenizer.batch_decode(outputs)
        return response[0]

    def extract_response(self: Self, text: str) -> str:
        # Fix the incorrect variable assignment when tag not found
        start_index = text.rfind("### Response:")
        
        if start_index == -1:
            start_index = text.rfind("<think>")
            if start_index == -1:
                return text.strip()
            return text[start_index + len("<think>"):].strip()
        else:
            return text[start_index + len("### Response:"):].strip()

    def generate_output(self: Self, question: str, max_turns: int = 5) -> dict:
        prompt = PROMPT_TEMPLATE.format(question=question)
        
        output = ""
        
        for turn in range(max_turns):
            print(f"\n--- Turn {turn+1} ---")
            response = self._run_ollama(prompt)
            
            with open("model_response.txt", "a") as f:
                f.write(response + "\n")
                f.write("-" * 50 + "\n")
            
            print(f"Model Response:\n{response}")
            response = self.extract_response(response)
            if "<think>" in response and "</think" not in response:
                response+= "</think>"
            
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
            
            ### Response:
            
            """
            embedded_context = (
                f"\n{self._start_result}\n{res}\n{self._end_result}\n{focus_reminder}"
            )
            
            prompt += f"\n{response}\n{embedded_context}\n"
            print(f"\n📎 Search results added to context. Continuing reasoning...\n")
        
            
        return self._compute_metrics(output)
