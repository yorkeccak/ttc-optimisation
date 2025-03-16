# rag_token_llm.py
# Rag Token LLM is an agentic RAG implementation that uses the Valyu API to search for information during the reasoning process.
# The reasoning model outputs <search> tokens when it needs to search for information.
# The search model then uses the Valyu API to search for the information and returns the results.
# The results are then passed to the reasoning model, which then uses the context to inform the answer.

from ..llm import Llm
from typing import Self

PROMPT_TEMPLATE = """
Answer the following question step-by-step:

"{question}"

If you know the answer then output it directly.

Otherwise, if you don't know the answer or think it is beyond then output a search query between {start_rag} and {end_rag} as follows:

{start_rag}[your search query here]{end_rag}

Use this format whenever you need to verify facts, obtain up-to-date information, or fill knowledge gaps.

Remember: Your output should be in the form of {start_rag}[your search query here]{end_rag} if you don't know or are uncertain/speculative about the answer. Otherwise, output the answer directly.
"""


class RagTokenLlm(Llm):
    def generate_output(self: Self, question: str, max_turns: int = 5) -> str:
        prompt = PROMPT_TEMPLATE.format(
            question=question,
            start_rag=self._start_rag,
            end_rag=self._end_rag,
        )

        output = ""

        for turn in range(max_turns):
            print(f"\n--- Turn {turn+1} ---")
            print(prompt)
            response = self._run_ollama(prompt, stop_tokens=[self._end_rag])

            if self._start_rag in response:
                response += self._end_rag

            output += response
            print(f"Model Response:\n{response}")
            search_query = self._extract_rag_query(response)

            if not search_query:
                print("\n[✅ No further searches required.]")
                print("\nFinal Response:\n")
                print(response)
                break

            print(f"\n[🔍 Searching for: '{search_query}']")
            res = self._run_valyu(search_query)
            embedded_context = f"\n{self._start_result}\n{res}\n{self._end_result}\n"
            prompt += f"\n{response}\n{embedded_context}\n"
            print(f"\n[✅ Search Results embedded into prompt.]\n")

        return output
