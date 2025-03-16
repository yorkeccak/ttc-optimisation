# rag_token_llm.py
# Rag Token LLM is an agentic RAG implementation that uses the Valyu API to search for information during the reasoning process.
# The reasoning model outputs <search> tokens when it needs to search for information.
# The search model then uses the Valyu API to search for the information and returns the results.
# The results are then passed to the reasoning model, which then uses the context to inform the answer.

from ..llm import Llm
from typing import Self

PROMPT_TEMPLATE = """
You are a helpful assistant that can search for information when needed. Answer the following question:

"{question}"

IMPORTANT: If you need to search for information, use EXACTLY this format:
{start_rag}your search query here{end_rag}

Examples:
Question: "What is the capital of France?"
If you need to search: {start_rag}capital city of France{end_rag}

Question: "Who won the 2024 Super Bowl?"
If you need to search: {start_rag}winner of 2024 Super Bowl{end_rag}

Rules:
1. ALWAYS use the EXACT format with {start_rag} and {end_rag}
2. Keep search queries short and focused
3. If you know the answer with certainty, respond directly without searching
4. Never include the tokens within your regular response text

Your turn - please answer the question above following these rules.
"""


class RagTokenLlm(Llm):
    def __init__(self, model: str) -> None:
        super().__init__(model)
        self._rag_enabled = True

    def generate_output(self: Self, question: str, max_turns: int = 5) -> str:
        prompt = PROMPT_TEMPLATE.format(
            question=question,
            start_rag=self._start_rag,
            end_rag=self._end_rag,
        )

        output = ""
        current_chunk = ""

        for turn in range(max_turns):
            print(f"\n--- Turn {turn+1} ---")
            print(prompt)

            # Stream the response and check for RAG tokens
            for chunk in self._run_ollama_stream(prompt, stop_tokens=[self._end_rag]):
                print(chunk, end="", flush=True)
                current_chunk += chunk

                # Check if we have a complete RAG query
                if self._start_rag in current_chunk and self._end_rag in current_chunk:
                    break

            # Ensure the RAG query is properly terminated
            if self._start_rag in current_chunk and not current_chunk.endswith(
                self._end_rag
            ):
                current_chunk += self._end_rag

            output += current_chunk
            search_query = self._extract_rag_query(current_chunk)

            if not search_query:
                print("\n[✅ No further searches required.]")
                print("\nFinal Response:\n")
                print(current_chunk)
                break

            print(f"\n[🔍 Searching for: '{search_query}']")
            res = self._run_valyu(search_query)
            embedded_context = f"\n{self._start_result}\n{res}\n{self._end_result}\n"
            prompt += f"\n{current_chunk}\n{embedded_context}\n"
            print(f"\n[✅ Search Results embedded into prompt.]\n")
            current_chunk = ""

        return output


myLLM = RagTokenLlm("deepseek-r1:1.5b")
print(myLLM.generate_output("What is the capital of France?"))
