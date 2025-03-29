# rag_token_llm.py
# Rag Token LLM is an agentic RAG implementation that uses the Valyu API to search for information during the reasoning process.
# The reasoning model outputs <search> tokens when it needs to search for information.
# The search model then uses the Valyu API to search for the information and returns the results.
# The results are then passed to the reasoning model, which then uses the context to inform the answer.

from ..llm import Llm
from typing import Self

PROMPT_TEMPLATE_OLD = """
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

PROMPT_TEMPLATE_NEW  = """ 
        You are a reasoning assistant with the ability to access an external knowledge source to help 
        you answer the user's question accurately. 
        Users Question is: {question}
        
        You have special tools that you MUST USE whenever you're unsure about something or when the question requires you to recall conceptual information, current events.
        To perform a search: output a relevant search query between {start_rag} and {end_rag} tokens like this: {start_rag}[your search query here]{end_rag} "
        Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format
        You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {max_turns}
        "Remember: You absolutely must search when the question requires factual information, specific data, current events, technical details,
        or any piece of knowledge you're confused or need to be sure about. This will help you get the right context and then you can answer the question."
        "- Use {start_rag} to request a web search and end with {end_rag}.\n"
        "- When done searching, continue your reasoning.\n\n"
        """


class RagTokenLlm(Llm):
    def __init__(self, model: str) -> None:
        super().__init__(model)
        self._rag_enabled = True

    def generate_output(self: Self, question: str, max_turns: int = 5) -> str:
        print("\n🤔 Initial Question:", question)
        prompt = PROMPT_TEMPLATE_NEW.format(
            question=question,
            start_rag=self._start_rag,
            end_rag=self._end_rag,
            max_turns=max_turns,
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
