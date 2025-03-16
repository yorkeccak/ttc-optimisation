# fine_tuned_llm.py
# Fine-tuned LLM is a more sophisticated version of rag_token_llm.py that uses a fine-tuned model to output search tokens instead of prompt engineering.

from ..llm import Llm
from typing import Self

class FineTunedLlm(Llm):
    def __init__(self, model: str) -> None:
        super().__init__(model)
        self._rag_enabled = True

    def generate_output(self: Self, question: str, max_turns: int = 5) -> str:
        return ""
