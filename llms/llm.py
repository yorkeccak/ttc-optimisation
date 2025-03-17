from abc import ABC, abstractmethod
import ollama
import re
from typing import Self, Generator
from valyu import Valyu
import tiktoken
from openai import OpenAI

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
        """
        response = self._valyu.context(
            query=query,
            search_type=search_type,
            max_num_results=max_num_results,
            max_price=max_price,
        )

        unfiltered_results = "\n".join([result.content for result in response.results])
        filtered_results = self._filter_results(
            previous_reasoning, query, unfiltered_results
        )
        return filtered_results

    def _filter_results(
        self: Self, previous_reasoning: str, query: str, unfiltered_results: str
    ) -> str:
        """
        Filter the results based on the previous reasoning.
        """
        client = OpenAI()

        prompt = WEB_SEARCH_PROMPT.format(
            prev_reasoning=previous_reasoning,
            search_query=query,
            document=unfiltered_results,
        )

        completion = client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": prompt}]
        )

        return completion.choices[0].message.content

    def _run_ollama(self: Self, prompt: str, stop_tokens=None) -> str:
        """
        Runs the model with the given prompt and returns the response.
        """
        response = ollama.generate(
            model=self._model,
            prompt=prompt,
            options={"stop": stop_tokens or []} if stop_tokens else {},
        )

        return response["response"]

    def _run_ollama_stream(
        self: Self, prompt: str, stop_tokens=None
    ) -> Generator[str, None, None]:
        """
        Runs the model with the given prompt and returns a generator that yields response chunks.
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
        """
        pattern = re.escape(self._start_rag) + r"(.*?)" + re.escape(self._end_rag)
        matches = re.findall(pattern, text, flags=re.DOTALL)
        return matches[-1].strip() if matches else None

    def _compute_metrics(self: Self, response: str) -> dict:
        """
        Computes the metrics for the model response, this includes no. of thinking tokens, full response tokens, no. of search queries, no. of search results.
        """
        # Count tokens in the response
        response_tokens = len(tokenizer.encode(response))

        # Count number of search queries
        search_queries = len(re.findall(re.escape(self._start_rag), response))

        # Count number of search results
        search_results = len(re.findall(re.escape(self._start_result), response))

        return {
            "response": response,
            "response_tokens": response_tokens,
            "search_queries": search_queries,
            "search_results": search_results,
        }

    @abstractmethod
    def generate_output(self: Self, question: str, max_turns: int = 5) -> str:
        pass
