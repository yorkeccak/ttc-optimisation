from abc import ABC, abstractmethod
import ollama
import re
from typing import Self, Generator
from valyu import Valyu


class Llm(ABC):
    def __init__(self: Self, model: str) -> None:
        super().__init__()
        self._model = model
        self._valyu = Valyu()
        self._rag_enabled = False
        self._start_rag = "<|begin_search_query|>"
        self._end_rag = "<|end_search_query|>"
        self._start_result = "<|begin_search_result|>"
        self._end_result = "<|end_search_result|>"

    def _run_valyu(
        self: Self,
        query: str,
        search_type: str = "all",
        max_num_results: int = 10,
        max_price: int = 10,
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

        return response

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
        E.g. <|begin_search_query|>What is the capital of France?<|end_search_query|> -> "What is the capital of France?"
        """
        pattern = re.escape(self._start_rag) + r"(.*?)" + re.escape(self._end_rag)
        matches = re.findall(pattern, text, flags=re.DOTALL)
        return matches[-1].strip() if matches else None

    @abstractmethod
    def generate_output(self: Self, question: str, max_turns: int = 5) -> str:
        pass
