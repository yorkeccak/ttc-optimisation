from abc import ABC, abstractmethod
import ollama
import re
from typing import Self
from valyu import Valyu


class Llm(ABC):
    def __init__(self: Self, model: str) -> None:
        super().__init__()
        self._model = model
        self._valyu = Valyu()
        self._start_rag = "<|begin_search_query|>"
        self._end_rag = "<|end_search_query|>"
        self._start_result = "<|begin_search_result|>"
        self._end_result = "<|end_search_result|>"

    def _run_valyu(self: Self, query: str) -> str:
        """
        Query the Valyu API to get relevant information for the model.
        """
        response = self._valyu.context(
            query=query,
            search_type="all",
            max_num_results=10,
            max_price=10,
        )

        return response.results[0].content

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
