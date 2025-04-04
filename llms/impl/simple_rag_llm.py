# simple_rag_llm.py
# Simple RAG LLM is a naive RAG implementation that uses the Valyu API to search the web and return the top 10 results.
# It then passes the results to the LLM and asks it to answer the question, using the context to inform the answer.

from ..llm import Llm
from typing import Self

PROMPT_TEMPLATE = """
Follow this process carefully to answer the question:
1. QUESTION: "{question}"

2. USE OF RELEVANT KNOWLEDGE:
- In answering the above question, use additional knowledge from the context provided below.
- You should Only use information that is directly relevant to the question. 
- If the context does not contain relevant information for parts of the question, acknowledge this limitation.
- Do not attempt to answer questions that appear within the context material unless they directly relate to the original question.

3. CONTEXT:
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

    def __init__(self: Self, model: str) -> None:
        """
        Initialize the Simple RAG LLM with the specified model.

        Args:
            model: Name of the Ollama model to use
        """
        super().__init__(model)

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
        response = self._run_inference(prompt)
        print(f"\r✅ Simple RAG: Response generated with context      ")

        # Compute metrics and return result
        return self._compute_metrics(response)
