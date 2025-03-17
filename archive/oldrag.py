from ..llm import Llm
from typing import Self

PROMPT_TEMPLATE = """
Follow this process carefully to answer the question:

1. QUESTION: "{question}"

2. KNOWLEDGE CHECK:
   - Consider carefully if you have COMPLETE and UP-TO-DATE knowledge to answer this question.
   - Be honest about your knowledge limitations - is the information factual, recent, or specific?
   - DO NOT make up information or provide speculative answers.

3. DECISION POINT:
   - If the question requires factual information, specific data, current events, technical details, 
     or any information you're not 100% certain about: YOU MUST SEARCH
   - Only answer directly for general knowledge, logical reasoning, or conceptual explanations you're absolutely certain about

4. ACTION:
   - To search: Output a concise, relevant search query between {start_rag} and {end_rag} tokens: 
     {start_rag}[your search query here]{end_rag}
   - This will provide you with search results beyond your knowledge base cutoff date which you can then use (you should assume that the search results provided are 100% correct and answer the question).
   - To answer directly: Provide a clear, accurate response based on what you know with certainty

Remember: The current date is March 13 2025, so using search you will be able to find up-to-date information until this date.  When in doubt, search rather than guess. It's better to verify information than to provide a potentially incorrect response." \
"""


class RagTokenLlm(Llm):
    def generate_output(self: Self, question: str, max_turns: int = 5) -> str:
        prompt = PROMPT_TEMPLATE.format(
            question=question,
            start_rag=self._start_rag,
            end_rag=self._end_rag,
        )

        output = ""
        original_question = question

        for turn in range(max_turns):
            print(f"\n--- Turn {turn+1} ---")
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
            print(f"\n[🔍 Search Results:]\n{res}")
            focus_reminder = f"""
Use the search results above to help answer the original question: "{original_question}"
Do not just summarize or explain the search results - use them to provide a direct answer to the question.
If you need additional information, you can search again using {self._start_rag}[your search query]{self._end_rag}
"""

            embedded_context = (
                f"\n{self._start_result}\n{res}\n{self._end_result}\n{focus_reminder}"
            )
            prompt += f"\n{response}\n{embedded_context}\n"
            print(f"\n[✅ Search Results embedded into prompt.]\n")

        return output
