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
    """
    No RAG LLM that generates responses without any external information.

    This implementation is the baseline approach:
    1. Take user question
    2. Generate response using only the LLM's internal knowledge
    """

    def __init__(self: Self, model: str) -> None:
        """
        Initialize the No RAG LLM with the specified model.

        Args:
            model: Name of the Ollama model to use
        """
        super().__init__(model)

    def generate_output(self: Self, question: str, max_turns: int = 5) -> dict:
        """
        Generate a response to the question using only the LLM (no external data).

        Args:
            question: The user's question
            max_turns: Maximum number of turns (unused in this implementation)

        Returns:
            Dictionary containing response and metrics
        """
        # Prepare prompt with just the question
        print(
            f"\r🧠 No RAG: Generating response with internal knowledge only...", end=""
        )
        prompt = PROMPT_TEMPLATE.format(question=question)

        # Generate response by streaming to properly time the thinking process
        response = ""
        for chunk in self._run_inference_stream(prompt):
            response += chunk

        print(f"\r✅ No RAG: Response generated using internal knowledge only       ")

        # Compute metrics and return result
        return self._compute_metrics(response, last_response=response)
