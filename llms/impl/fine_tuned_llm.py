# fine_tuned_llm.py
# Fine-tuned LLM is a more sophisticated version of rag_token_llm.py that uses a fine-tuned model to output search tokens instead of prompt engineering.

from ..llm import Llm
from typing import Self
import torch
import gc
from unsloth import FastLanguageModel
from transformers import AutoTokenizer
from peft import PeftModel
from transformers import StoppingCriteria, StoppingCriteriaList

PROMPT_TEMPLATE = """
Below is an instruction that describes a task, paired with an input that provides further context.

Write a response that appropriately completes the request.

Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a scientific expert with advanced knowledge in analytical reasoning, problem-solving, and quantitative analysis across various disciplines, including mathematics, physics, biology, chemistry, and engineering. Please answer the following question.

### Question:
{question}

### Response:

"""

# PROMPT_TEMPLATE = """
# QUESTION: {question}

# Answer this question accurately. If you're uncertain about any facts:
# 1. IMMEDIATELY output a search query between {start_rag} and {end_rag} tags
# 2. Wait for search results between {start_result} and {end_result}
# 3. Use these results to complete your answer

# EXAMPLES:
# Q: "Who is the current CEO of OpenAI?"
# {start_rag}current CEO of OpenAI 2025{end_rag}
# {start_result}Sam Altman returned as OpenAI CEO in November 2023 and continues to serve in this role as of March 2025.{end_result}
# The current CEO of OpenAI is Sam Altman.

# Q: "What is the population of Tokyo?"
# {start_rag}current population of Tokyo 2025{end_rag}
# {start_result}Tokyo's population is approximately 13.96 million as of January 2025.{end_result}
# Tokyo has a population of approximately 13.96 million people.

# Today's date: April 1, 2025. Be direct and concise.
# """


class FineTunedLlm(Llm):
    def __init__(self: Self) -> None:
        super().__init__("")
        self._rag_enabled = True
        self._start_rag = "<SEARCH>"
        self._end_rag = "</SEARCH>"
        self._start_result = "<RESULT>"
        self._end_result = "</RESULT>"

        # ===================
        # = LOAD BASE MODEL =
        # ===================

        max_seq_length = 2048
        dtype = None
        load_in_4bit = True

        base_model, _ = FastLanguageModel.from_pretrained(
            model_name = "unsloth/DeepSeek-R1-Distill-Llama-8B",
            max_seq_length = max_seq_length,
            dtype = dtype,
            load_in_4bit = load_in_4bit,
        )

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

        # Exclude existing tokens
        new_tokens = list(special_tokens - set(_.vocab.keys()))

        if new_tokens:
            print("Addding new tokens... ")

            # Add new tokens to tokenizer and model using add_new_tokens()
            self.add_new_tokens(base_model, _, new_tokens=new_tokens)

            # Verify the additions
            print(f"Added {len(new_tokens)} new tokens to the tokenizer and model! ")
        else:
            print("No new tokens added. ")

        # =============================
        # = ADD ADAPTER TO BASE MODEL =
        # =============================

        self.model = PeftModel.from_pretrained(base_model, "canertugrul/DeepSeek-R1-Distill-Llama-8B-Tool-Use-Adapter", cache_dir="/cs/student/projects1/2021/jadhwani/.cache/huggingface/hub")
        FastLanguageModel.for_inference(self.model)

        # ==================
        # = LOAD TOKENIZER =
        # ==================

        self.tokenizer = AutoTokenizer.from_pretrained("canertugrul/DeepSeek-R1-Distill-Llama-8B-Tool-Use-Tokenizer", cache_dir="/cs/student/projects1/2021/jadhwani/.cache/huggingface/hub")

    def add_new_tokens(self, model, tokenizer, new_tokens=[], method="mean", interpolation=0.5):
        assert isinstance(new_tokens, (list, tuple))
        assert len(new_tokens) > 0
        assert method in ["mean", "interpolation"]
        assert 0 <= interpolation <= 1
        overlapping_tokens = set(new_tokens) & set(tokenizer.vocab.keys())
        
        if overlapping_tokens:
            print(f"Unsloth: Skipping overlapping tokens: {list(overlapping_tokens)}")
            new_tokens = [x for x in new_tokens if x not in overlapping_tokens]

        # Add new tokens to tokenizer
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

        FastLanguageModel.for_inference(self.model)
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        stop_token_id = self.tokenizer.convert_tokens_to_ids(self._end_rag)

        stopping_criteria = StoppingCriteriaList([
            StopOnTokens([stop_token_id])
        ])

        outputs = self.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=2048,
            use_cache=True,
            stopping_criteria=stopping_criteria
        )

        response = self.tokenizer.batch_decode(outputs)
        return response[0]

    def generate_output(self: Self, question: str, max_turns: int = 5) -> dict:
        prompt = PROMPT_TEMPLATE.format(question=question)
        output = self._run_ollama(prompt)
        return self._compute_metrics(output)
