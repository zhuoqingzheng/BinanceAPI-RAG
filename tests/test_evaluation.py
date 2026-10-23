"""Tests for the eval's matching and metrics. No database or API calls."""

import json

import pytest

from rag.config import PROJECT_ROOT
from rag.evaluation import (
    Question,
    Section,
    first_hit_rank,
    format_report,
    hit_rate,
    is_match,
    load_questions,
    mean_reciprocal_rank,
)
from rag.search import SearchResult


def _result(doc_path: str, heading_path: str) -> SearchResult:
    return SearchResult(0, doc_path, heading_path, "https://example.com", "text", 0.0)


LOT_SIZE = Section("filters.md", "LOT_SIZE")


def test_match_needs_the_same_file_and_heading():
    assert is_match(
        _result("filters.md", "Filters > Symbol filters > LOT_SIZE"), LOT_SIZE
    )
    assert not is_match(
        _result("errors.md", "Filters > Symbol filters > LOT_SIZE"), LOT_SIZE
    )
    assert not is_match(
        _result("filters.md", "Filters > Symbol filters > MARKET_LOT_SIZE"), LOT_SIZE
    )


def test_match_includes_subsections():
    section = Section("faqs/stp_faq.md", "Self Trade Prevention (STP) FAQ")
    assert is_match(
        _result(
            "faqs/stp_faq.md",
            "Self Trade Prevention (STP) FAQ > What is a Trade Group Id?",
        ),
        section,
    )


def test_first_hit_rank_counts_from_1_and_accepts_any_expected_section():
    results = [
        _result("errors.md", "Error codes > -1020 UNSUPPORTED_OPERATION"),
        _result("rest-api.md", "Trading endpoints > New order (TRADE)"),
        _result("filters.md", "Filters > Symbol filters > LOT_SIZE"),
    ]
    expected = (Section("web-socket-api.md", "Place new order (TRADE)"), LOT_SIZE)
    assert first_hit_rank(results, expected) == 3
    assert first_hit_rank(results, (Section("enums.md", "STP Modes"),)) is None


def test_hit_rate_and_mrr():
    ranks = [1, 2, 6, None]
    assert hit_rate(ranks, 1) == 0.25
    assert hit_rate(ranks, 5) == 0.5
    assert mean_reciprocal_rank(ranks, 10) == pytest.approx((1 + 1 / 2 + 1 / 6 + 0) / 4)
    # Rank 6 is outside the top 5, so it scores 0 for MRR@5.
    assert mean_reciprocal_rank(ranks, 5) == pytest.approx((1 + 1 / 2 + 0 + 0) / 4)


def test_load_questions_rejects_a_question_without_expected_sections(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text(
        json.dumps({"id": "q1", "type": "exact", "question": "?", "expected": []})
    )
    with pytest.raises(ValueError, match="no expected sections"):
        load_questions(path)


def test_load_questions_rejects_duplicate_ids(tmp_path):
    line = json.dumps(
        {
            "id": "q1",
            "type": "exact",
            "question": "?",
            "expected": [{"file": "a.md", "heading": "A"}],
        }
    )
    path = tmp_path / "questions.jsonl"
    path.write_text(f"{line}\n{line}\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_questions(path)


def test_questions_file_is_valid():
    questions = load_questions(PROJECT_ROOT / "eval" / "questions.jsonl")
    assert len(questions) >= 30
    assert {q.type for q in questions} <= {
        "exact",
        "how_to",
        "concept",
        "compare",
        "paraphrase",
    }


def test_report_shows_ranks_and_summary():
    questions = [
        Question("q1", "exact", "What is LOT_SIZE?", (LOT_SIZE,)),
        Question("q2", "paraphrase", "Why was my order rejected?", (LOT_SIZE,)),
    ]
    ranks = {"vector": [1, 2], "keyword": [1, None], "hybrid": [3, 1]}
    report = format_report(questions, ranks, k=1)
    assert "q2   paraphrase     2   -    1  Why was my order rejected?" in report
    assert "keyword      50%    50%    0.50" in report
