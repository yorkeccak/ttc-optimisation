from abc import ABC, abstractmethod
import ollama
import re
from typing import Self, Generator
from valyu import Valyu
import tiktoken
import cohere
import os

# Initialize tokenizer for token counting
tokenizer = tiktoken.get_encoding("cl100k_base")


WEB_SEARCH_PROMPT = """**Task Instruction:**

You are tasked with reading and analyzing web pages based on the following inputs: **Previous Reasoning Steps**, **Current Search Query**, and **Searched Web Pages**. Your objective is to extract relevant and helpful information for **Current Search Query** from the **Searched Web Pages** and seamlessly integrate this information into the **Previous Reasoning Steps** to continue reasoning for the original question.

**Guidelines:**

1. **Analyze the Searched Web Pages:**
- Carefully review the content of each searched web page.
- Identify factual information that is relevant to the **Current Search Query** and can aid in the reasoning process for the original question.

2. **Extract Relevant Information:**
- Select the information from the Searched Web Pages that directly contributes to advancing the **Previous Reasoning Steps**.
- Ensure that the extracted information is accurate and relevant.

3. **Output Format:**
- **If the web pages provide helpful information for current search query:** Present the information beginning with `**Final Information**` as shown below.
**Final Information**

[Helpful information]

- **If the web pages do not provide any helpful information for current search query:** Output the following text.

**Final Information**

No helpful information found.

**Inputs:**
- **Previous Reasoning Steps:**  
{prev_reasoning}

- **Current Search Query:**  
{search_query}

- **Searched Web Pages:**  
{document}

Now you should analyze each web page and find helpful information based on the current search query "{search_query}" and previous reasoning steps.
"""


class Llm(ABC):
    def __init__(self: Self, model: str) -> None:
        super().__init__()
        self._model = model
        self._valyu = Valyu()
        self._rag_enabled = False
        self._start_rag = "<begin_search_query>"
        self._end_rag = "</end_search_query>"
        self._start_result = "<begin_search_result>"
        self._end_result = "</end_search_result>"
        self._filtered_search_results = ""  # Store filtered search results

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
            f"\r✅ Found {len(response.results)} search results                      "
        )

        # Store raw results for logging
        raw_results = "\n".join([result.content for result in response.results])

        # Filter results
        print(f"\r🧠 Filtering search results with Cohere...", end="")
        filtered_results = self._filter_results(previous_reasoning, query, raw_results)
        print(f"\r✅ Search results filtered                    ")

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

    def _run_ollama(self: Self, prompt: str, stop_tokens=None) -> str:
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

    def _run_ollama_stream(
        self: Self, prompt: str, stop_tokens=None
    ) -> Generator[str, None, None]:
        """
        Runs the model with the given prompt and returns a generator that yields response chunks.

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

        for chunk in stream:
            yield chunk["message"]["content"]

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

    def _compute_metrics(self: Self, response: str) -> dict:
        """
        Computes the metrics for the model response, this includes no. of thinking tokens,
        full response tokens, no. of search queries, no. of search results, and filtered search results.

        Args:
            response: Model response text

        Returns:
            Dictionary containing metrics and model response
        """
        # Count tokens in the response
        response_tokens = len(tokenizer.encode(response))

        # Count number of search queries
        search_queries = len(re.findall(re.escape(self._start_rag), response))

        # Count number of search results
        search_results = len(re.findall(re.escape(self._start_result), response))

        # Include filtered search results if available
        metrics = {
            "response": response,
            "response_tokens": response_tokens,
            "search_queries": search_queries,
            "search_results": search_results,
        }

        # Add filtered search results if available
        if self._filtered_search_results:
            metrics["filtered_search_results"] = self._filtered_search_results

        return metrics

    @abstractmethod
    def generate_output(self: Self, question: str, max_turns: int = 5) -> str:
        pass
