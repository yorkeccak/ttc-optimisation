from abc import ABC, abstractmethod
import ollama
import re
import time
from typing import Self, Generator
from valyu import Valyu
import tiktoken
import cohere
import os

WEB_SEARCH_PROMPT = """**Task Instruction: EXTRACTION ONLY, DO NOT ANSWER THE QUESTION**

You are an information extraction tool ONLY. Your job is to extract relevant facts from web pages WITHOUT answering, reasoning about, or addressing the original question directly.

**IMPORTANT: You are NOT to provide answers, conclusions, opinions, or analyses. You are ONLY extracting information.**

Your inputs:
- **Previous Reasoning Steps:** {prev_reasoning}
- **Current Search Query:** {search_query}
- **Searched Web Pages:** {document}

**Strict Guidelines:**
1. Extract ONLY factual information from the web pages that is relevant to the search query
2. Present information as brief, bullet-point style facts
3. DO NOT integrate these facts into the reasoning steps
4. DO NOT attempt to answer the original question
5. DO NOT provide your own analysis, recommendations, or conclusions
6. DO NOT attempt to finish the user's reasoning

**Output Format:**
- Begin with "**Extracted Facts:**" followed by bullet points of ONLY factual information from the search results
- If no relevant information found, output: "**Extracted Facts:** No relevant factual information found."

Your extraction should be concise, factual, and detached from any attempt to solve the user's original problem.
"""


class Llm(ABC):
    def __init__(self: Self, model: str) -> None:
        super().__init__()
        self._model = model
        self._valyu = Valyu()
        self._start_rag = "<begin_search_query>"
        self._end_rag = "</end_search_query>"
        self._start_result = "<begin_search_result>"
        self._end_result = "</end_search_result>"
        self._filtered_search_results = ""
        self._tokenizer = tiktoken.get_encoding("cl100k_base")
        self._thinking_tags = ("<think>", "</think>")
        self._thinking_start = None
        self._in_thinking = False
        self._thinking_times = []  # Track multiple thinking session times

    def _run_valyu(
        self: Self,
        query: str,
        search_type: str = "web",
        max_num_results: int = 5,
        max_price: int = 10,
        previous_reasoning: str = "",
    ) -> str:
        """
        Query the Valyu API to get relevant information for the model.

        Args:
            query: The search query
            search_type: Type of search ('web', 'news', etc.)
            max_num_results: Maximum number of results to return
            max_price: Maximum price in credits to spend
            previous_reasoning: Any previous reasoning steps

        Returns:
            Filtered search results relevant to the query
        """
        print(
            f"\r🔍 Searching with query: {query[:50]}{'...' if len(query) > 50 else ''}",
            end="",
        )
        response = self._valyu.context(
            query=query,
            search_type=search_type,
            max_num_results=max_num_results,
            max_price=max_price,
        )
        print(
            f"\r✅ Found {len(response.results)} search results"
        )

        # Store raw results for logging
        raw_results = "\n".join([result.content for result in response.results])

        # Filter results
        print(f"\r🧠 Filtering search results with Cohere...", end="")
        filtered_results = self._filter_results(previous_reasoning, query, raw_results)
        print(f"\r✅ Search results filtered ")

        # Store the filtered results for later retrieval
        self._filtered_search_results = filtered_results

        return filtered_results

    def _filter_results(
        self: Self, previous_reasoning: str, query: str, unfiltered_results: str
    ) -> str:
        """
        Filter the results based on the previous reasoning using Cohere instead of OpenAI.

        Args:
            previous_reasoning: The reasoning steps so far
            query: The current search query
            unfiltered_results: Raw search results from Valyu

        Returns:
            Filtered results containing only relevant information
        """
        # Get Cohere API key
        cohere_api_key = os.getenv("COHERE_API_KEY")
        if not cohere_api_key:
            raise ValueError("COHERE_API_KEY environment variable is not set")

        co = cohere.Client(cohere_api_key)

        prompt = WEB_SEARCH_PROMPT.format(
            prev_reasoning=previous_reasoning,
            search_query=query,
            document=unfiltered_results,
        )

        response = co.chat(model="command-a-03-2025", message=prompt, temperature=0.0)

        return response.text

    def _run_inference(self: Self, prompt: str, stop_tokens=None) -> str:
        """
        Runs the model with the given prompt and returns the response.

        Args:
            prompt: The prompt to send to the model
            stop_tokens: Optional list of stop tokens

        Returns:
            Model response text
        """
        print(f"\r🤖 Generating response with {self._model}...", end="")
        response = ollama.generate(
            model=self._model,
            prompt=prompt,
            options={"stop": stop_tokens or []} if stop_tokens else {},
        )
        print(f"\r✅ Response generated                          ")

        return response["response"]

    def _run_inference_stream(
        self: Self, prompt: str, stop_tokens=None
    ) -> Generator[str, None, None]:
        """
        Runs the model with the given prompt and returns a generator that yields response chunks.
        Also times the thinking process by detecting <think> and </think> tags.

        Args:
            prompt: The prompt to send to the model
            stop_tokens: Optional list of stop tokens

        Returns:
            Generator yielding response chunks
        """
        stream = ollama.chat(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            options={"stop": stop_tokens or []} if stop_tokens else {},
        )

        buffer = ""
        try:
            for chunk in stream:
                text_chunk = chunk["message"]["content"]
                buffer += text_chunk
                # Check for thinking start and end in the buffer
                if not self._in_thinking and "<think>" in buffer:
                    self._in_thinking = True
                    self._thinking_start = time.time()
                    print(f"\r🧠 Thinking started...", end="")

                if self._in_thinking and "</think>" in buffer:
                    self._in_thinking = False
                    thinking_time = time.time() - self._thinking_start
                    self._thinking_times.append(thinking_time)
                    print(
                        f"\r✅ Thinking completed in {thinking_time:.2f} seconds",
                        end="",
                    )
                    buffer = ""  # Reset buffer after capturing a thinking session
                yield text_chunk
        finally:
            print("\r✅ Stream ended, cleaning up...", end="")
            # Handle the case where we search
            if self._in_thinking:
                self._in_thinking = False
                thinking_time = time.time() - self._thinking_start
                self._thinking_times.append(thinking_time)
                print(f"\r✅ Thinking captured ({thinking_time:.2f} seconds)", end="")

    def _extract_rag_query(self: Self, text: str) -> str | None:
        """
        Extracts the RAG query from the model output.
        E.g. <begin_search_query>What is the capital of France?<end_search_query> -> "What is the capital of France?"

        Args:
            text: Text containing possible RAG queries

        Returns:
            Extracted query or None if not found
        """
        pattern = re.escape(self._start_rag) + r"(.*?)" + re.escape(self._end_rag)
        matches = re.findall(pattern, text, flags=re.DOTALL)
        return matches[-1].strip() if matches else None

    def _compute_metrics(self: Self, response: str, last_response: str) -> dict:
        """
        Computes the metrics for the model response, this includes no. of thinking tokens,
        full response tokens, no. of search queries, no. of search results, and filtered search results.

        Args:
            response: Model response text

        Returns:
            Dictionary containing metrics and model response
        """
        # Count tokens in the response
        response_tokens = len(self._tokenizer.encode(response))

        # Count number of search queries
        search_queries = len(re.findall(re.escape(self._start_rag), response))

        # Count number of search results
        search_results = len(re.findall(re.escape(self._start_result), response))

        total_thinking_time = sum(self._thinking_times) if self._thinking_times else 0
        
        last_think = last_response.rfind(self._thinking_tags[1])
        if last_think != -1:
            last_response = last_response[last_think + len(self._thinking_tags[1]): ]

        print("Last response", last_response)

        # Include filtered search results if available
        metrics = {
            "response": response,
            "last_response": last_response,
            "response_tokens": response_tokens,
            "thinking_time": total_thinking_time,
            "thinking_time_sessions": self._thinking_times.copy(),
            "search_queries": search_queries,
            "search_results": search_results,
        }

        # Add filtered search results if available
        if self._filtered_search_results:
            metrics["filtered_search_results"] = self._filtered_search_results

        # Reset filtered search results
        self._filtered_search_results = ""

        # Reset thinking times for next query
        self._thinking_times = []

        return metrics

    @abstractmethod
    def generate_output(self: Self, question: str, max_turns: int = 5) -> dict:
        pass
