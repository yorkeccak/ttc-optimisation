# no_rag_llm.py
# No RAG LLM is a simple LLM that does not use any RAG.
# It simply uses the LLM to answer the question.

from ..llm import Llm
from typing import Self

PROMPT_TEMPLATE = """
You are a knowledgeable research assistant tasked with providing accurate, well-structured answers. Your goal is to answer ONLY the following question:

"{question}"

Do not try to answer any other questions or topics - focus exclusively on the question above.
"""


class NoRagLlm(Llm):
    def generate_output(self: Self, question: str, max_turns: int = 5) -> str:
        prompt = PROMPT_TEMPLATE.format(question=question)
        return self._run_ollama(prompt)
