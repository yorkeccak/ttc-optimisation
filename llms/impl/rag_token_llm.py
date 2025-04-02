# rag_token_llm.py
# Rag Token LLM is an agentic RAG implementation that uses the Valyu API to search for information during the reasoning process.
# The reasoning model outputs <search> tokens when it needs to search for information.
# The search model then uses the Valyu API to search for the information and returns the results.
# The results are then passed to the reasoning model, which then uses the context to inform the answer.

from ..llm import Llm
from typing import Self


PROMPT_TEMPLATE_NEW = """
QUESTION: {question}

Answer this question accurately. If you're uncertain about any facts:
1. IMMEDIATELY output a search query between {start_rag} and {end_rag} tags
2. Wait for search results between {start_result} and {end_result}
3. Use these results to complete your answer

EXAMPLES:
Q: "Who is the current CEO of OpenAI?"
{start_rag}current CEO of OpenAI 2025{end_rag}
{start_result}Sam Altman returned as OpenAI CEO in November 2023 and continues to serve in this role as of March 2025.{end_result}
The current CEO of OpenAI is Sam Altman.

Q: "What is the population of Tokyo?"
{start_rag}current population of Tokyo 2025{end_rag}
{start_result}Tokyo's population is approximately 13.96 million as of January 2025.{end_result}
Tokyo has a population of approximately 13.96 million people.

Today's date: April 1, 2025. Be direct and concise.
"""


class RagTokenLlm(Llm):
    def __init__(self, model: str) -> None:
        super().__init__(model)
        self._rag_enabled = True
        print(PROMPT_TEMPLATE_NEW.format(
            question='test',
            start_rag=self._start_rag,
            end_rag=self._end_rag,
            start_result=self._start_result,
            end_result=self._end_result,
        ))

    def generate_output(self: Self, question: str, max_turns: int = 5) -> str:
        print("\n🤔 Initial Question:", question)
        prompt = PROMPT_TEMPLATE_NEW.format(
            question=question,
            start_rag=self._start_rag,
            end_rag=self._end_rag,
            start_result=self._start_result,
            end_result=self._end_result,
        )
        # prompt = PROMPT_TEMPLATE.format(
        #     question=question,
        #     start_rag=self._start_rag,
        #     end_rag=self._end_rag,
        #     max_turns=max_turns,
        # )

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

            focus_reminder = f"""
Use the search results above to help answer the original question: "{question}"
Do not just summarize or explain the search results - use them to provide a direct answer to the question.
If you need additional information, you can search again using {self._start_rag}[your search query]{self._end_rag}
"""

            embedded_context = (
                f"\n{self._start_result}\n{res}\n{self._end_result}\n{focus_reminder}"
            )
            prompt += f"\n{current_chunk}\n{embedded_context}\n"
            print(f"\n📎 Search results added to context. Continuing reasoning...\n")
            current_chunk = ""

        return self._compute_metrics(output)
