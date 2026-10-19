"""Answer a question from the retrieved chunks, citing them by number."""

import re
from dataclasses import dataclass
from typing import Protocol

from rag.config import Settings
from rag.search import SearchResult

INSTRUCTIONS = """\
You answer questions about the Binance Spot API using only the numbered sources provided.

- Cite the sources you used by number in square brackets, e.g. [1] or [2][3].
- If the sources don't answer the question, say so plainly. Don't use outside knowledge.
- Be concise. Quote endpoint paths, parameter names and error codes exactly as written.
"""


class LLM(Protocol):
    """Anything that turns instructions + a prompt into text, so providers can be swapped."""

    def complete(self, instructions: str, prompt: str) -> str: ...


class OpenAILLM:
    """Chat completions from the OpenAI API."""

    def __init__(self, api_key: str, model: str):
        from openai import OpenAI

        # The client retries rate-limit and server errors itself.
        self._client = OpenAI(api_key=api_key, max_retries=5)
        self._model = model

    def complete(self, instructions: str, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[SearchResult]  # sources[0] is cited as [1] in the text


def build_prompt(question: str, results: list[SearchResult]) -> str:
    """The sources, numbered from 1, then the question."""
    sources = "\n\n".join(
        f"[{number}] {r.doc_path} > {r.heading_path}\n{r.content}"
        for number, r in enumerate(results, start=1)
    )
    return f"Sources:\n\n{sources}\n\nQuestion: {question}"


def answer_question(question: str, results: list[SearchResult], llm: LLM) -> Answer:
    if not results:
        return Answer("No matching sections found. Has `rag ingest` been run?", [])
    return Answer(llm.complete(INSTRUCTIONS, build_prompt(question, results)), results)


def cited_sources(answer: Answer) -> list[tuple[int, SearchResult]]:
    """The (number, source) pairs the answer actually cites, in number order."""
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer.text)}
    return [(n, answer.sources[n - 1]) for n in sorted(cited) if 1 <= n <= len(answer.sources)]


def get_llm(settings: Settings) -> LLM:
    if not settings.openai_api_key:
        raise SystemExit("OPENAI_API_KEY is not set. Add it to .env.")
    return OpenAILLM(settings.openai_api_key, settings.answer_model)
