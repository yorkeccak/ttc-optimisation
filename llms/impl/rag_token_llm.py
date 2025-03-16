# rag_token_llm.py
# Rag Token LLM is an agentic RAG implementation that uses the Valyu API to search for information during the reasoning process.
# The reasoning model outputs <search> tokens when it needs to search for information.
# The search model then uses the Valyu API to search for the information and returns the results.
# The results are then passed to the reasoning model, which then uses the context to inform the answer.

from ..llm import Llm
from typing import Self

PROMPT_TEMPLATE = """
Answer the following question step-by-step. 
If you don't know the answer or think it is beyond then place your search query between these tokens {start_rag}[your search query here]{end_rag}
If you dont know then immediately admit it and dont try to reason.
OR if you know the answer then output it directly.
Use this format whenever you need to verify facts, obtain up-to-date information, or fill knowledge gaps.
This helps identify exactly what information you need to search for.
\nQuestion: {question}
\nAnswer
Remember, your output should be in the form of {start_rag}[your search query here]{end_rag} if you don't know or are uncertain/speculative about the answer. Otherwise, provide the answer directly.
    """


class RagTokenLlm(Llm):
    def __init__(self, model: str) -> None:
        super().__init__(model)
        self._rag_enabled = True

    def generate_output(self: Self, question: str, max_turns: int = 5) -> str:
        print("\n🤔 Initial Question:", question)
        prompt = PROMPT_TEMPLATE.format(
            question=question,
            start_rag=self._start_rag,
            end_rag=self._end_rag,
        )

        output = ""
        current_chunk = ""

        for turn in range(max_turns):
            print(f"\n📝 Turn {turn+1}/{max_turns} ---")
            print("\n🤖 Model thinking...")

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
                print("\n✅ Model provided direct answer (no search needed)")
                print("\n📊 Final Response:")
                print("=" * 50)
                print(current_chunk)
                print("=" * 50)
                break

            print(f"\n🔍 Searching: '{search_query}'")
            res = self._run_valyu(search_query)
            print("\n📚 Search Results:")
            print("-" * 50)
            print(res)
            print("-" * 50)

            embedded_context = f"\n{self._start_result}\n{res}\n{self._end_result}\n"
            prompt += f"\n{current_chunk}\n{embedded_context}\n"
            print(f"\n📎 Search results added to context. Continuing reasoning...\n")
            current_chunk = ""

        return output