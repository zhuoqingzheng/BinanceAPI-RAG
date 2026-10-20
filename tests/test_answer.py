"""Tests for the answer step. A fake LLM stands in for the API, so these cost nothing."""

from rag.answer import (
    INSTRUCTIONS,
    Answer,
    answer_question,
    build_prompt,
    cited_sources,
)
from rag.search import SearchResult


def _result(n: int) -> SearchResult:
    return SearchResult(
        chunk_id=n,
        doc_path="rest-api.md",
        heading_path=f"Trading endpoints > Endpoint {n}",
        url=f"https://example.com/rest-api.md#endpoint-{n}",
        content=f"Content of endpoint {n}.",
        score=1.0 / n,
    )


class FakeLLM:
    def __init__(self, reply: str):
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def complete(self, instructions: str, prompt: str) -> str:
        self.calls.append((instructions, prompt))
        return self.reply


def test_prompt_numbers_sources_from_1_and_ends_with_the_question():
    prompt = build_prompt("How do I place an order?", [_result(1), _result(2)])
    assert (
        "[1] rest-api.md > Trading endpoints > Endpoint 1\nContent of endpoint 1."
        in prompt
    )
    assert (
        "[2] rest-api.md > Trading endpoints > Endpoint 2\nContent of endpoint 2."
        in prompt
    )
    assert prompt.endswith("Question: How do I place an order?")


def test_answer_sends_instructions_and_prompt_to_the_llm():
    llm = FakeLLM("Use POST /api/v3/order [1].")
    results = [_result(1), _result(2)]
    answer = answer_question("How do I place an order?", results, llm)
    assert answer == Answer("Use POST /api/v3/order [1].", results)
    assert llm.calls == [
        (INSTRUCTIONS, build_prompt("How do I place an order?", results))
    ]


def test_no_results_means_no_llm_call():
    llm = FakeLLM("should not be used")
    answer = answer_question("anything", [], llm)
    assert answer.sources == []
    assert llm.calls == []


def test_cited_sources_lists_only_cited_numbers_in_order():
    results = [_result(1), _result(2), _result(3)]
    answer = Answer("See [3] and [1][3]. Ignore [7].", results)
    assert cited_sources(answer) == [(1, results[0]), (3, results[2])]
