from ..llm import Llm
from typing import Self

PROMPT_TEMPLATE = """
You are a knowledgeable research assistant tasked with providing accurate, well-structured answers. Your goal is to answer ONLY the following question:

"{question}"

Do not try to answer any other questions or topics - focus exclusively on the question above.

Below is relevant context to help inform your response. Only use information that is directly relevant to the question. If the context does not contain relevant information for parts of the question, acknowledge this limitation.

<context>
{context}
</context>

Remember: Stay focused only on answering the original question about "{question}". Do not attempt to answer questions that appear within the context material unless they directly relate to the original question.
"""


class SimpleRagLlm(Llm):
    def generate_output(self: Self, question: str, max_turns: int = 5) -> str:
        context = self._run_valyu(question)
        prompt = PROMPT_TEMPLATE.format(question=question, context=context)
        return self._run_ollama(prompt)
