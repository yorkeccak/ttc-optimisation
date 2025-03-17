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
        self._start_rag = "<search_query>"
        self._end_rag = "</search_query>"
        self._start_result = "<search_result>"
        self._end_result = "</search_result>"

    def _run_valyu(self: Self, query: str) -> str:
        for _ in range(10):
            try:
                response = self._valyu.context(
                    query=query,
                    search_type="web",
                    max_num_results=10,
                    max_price=10,
                )
                # response.results[0].content
                return "\n".join([result.content for result in response.results])
            except:
                query+='.'
                print('CRASH')
        exit()

    def _run_ollama(self: Self, prompt: str, stop_tokens=None) -> str:
        response = ollama.generate(
            model=self._model,
            prompt=prompt,
            options={"num_ctx": 8192, "stop": stop_tokens or []} if stop_tokens else {"num_ctx": 8192},
        )
        return response["response"]

    def _extract_rag_query(self: Self, text: str) -> str | None:
        pattern = re.escape(self._start_rag) + r"(.*?)" + re.escape(self._end_rag)
        matches = re.findall(pattern, text, flags=re.DOTALL)
        return matches[-1].strip() if matches else None

    @abstractmethod
    def generate_output(self: Self, question: str, max_turns: int = 5) -> str:
        pass
