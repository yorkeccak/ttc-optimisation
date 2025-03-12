from ..llm import Llm
from typing import Self


class FineTunedLlm(Llm):
    def generate_output(self: Self, question: str, max_turns: int = 5) -> str:
        return ""
