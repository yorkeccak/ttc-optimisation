# simple_rag_llm.py
# Simple RAG LLM is a naive RAG implementation that uses the Valyu API to search the web and return the top 10 results.
# It then passes the results to the LLM and asks it to answer the question, using the context to inform the answer.

import time
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
    """
    Simple RAG LLM that performs search before generating a response.

    This implementation follows a standard RAG approach:
    1. Take user question
    2. Search for relevant information using Valyu API
    3. Include search results as context in the prompt
    4. Generate a response with the LLM using this context
    """

    def __init__(self, model: str) -> None:
        """
        Initialize the Simple RAG LLM with the specified model.

        Args:
            model: Name of the Ollama model to use
        """
        super().__init__(model)
        self._rag_enabled = True

    def generate_output(self: Self, question: str) -> dict:
        """
        Generate a response to the question using Simple RAG approach.

        Args:
            question: The user's question

        Returns:
            Dictionary containing response and metrics
        """
        # Search for context using Valyu API
        print(f"\r🔍 Simple RAG: Searching for context for question...", end="")
        context = self._run_valyu(question)
        print(f"\r✅ Simple RAG: Retrieved context for question        ")

        # Prepare prompt with question and context
        prompt = PROMPT_TEMPLATE.format(question=question, context=context)

        # Generate response using the LLM
        print(f"\r🧠 Simple RAG: Generating response with context...", end="")
        
        response = self._run_ollama(prompt)

        print(f"\r✅ Simple RAG: Response generated with context      ")

        # Compute metrics and return result
        return self._compute_metrics(response)
