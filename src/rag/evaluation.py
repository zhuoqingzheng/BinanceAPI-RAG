"""Retrieval eval: how high does each search mode rank the correct section?

Each question in eval/questions.jsonl lists the doc sections that answer it.
For every question and mode we find the rank of the first correct chunk, then
summarise the ranks as hit@k and MRR.
"""

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from rag import search
from rag.search import SearchResult

MODES = ("vector", "keyword", "hybrid")
# Results fetched per question and mode. The per-question table shows ranks up
# to here, so near misses (rank 6) differ from total misses; the metrics only
# count the top k.
DEPTH = 10


@dataclass(frozen=True)
class Section:
    file: str  # doc_path, e.g. "filters.md"
    heading: str  # one heading title, e.g. "LOT_SIZE"


@dataclass(frozen=True)
class Question:
    id: str
    type: str  # exact, how_to, concept, compare or paraphrase
    question: str
    expected: tuple[Section, ...]


def load_questions(path: Path) -> list[Question]:
    """Read one question per line. Unknown fields (e.g. must_include) are ignored."""
    questions = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        data = json.loads(line)
        expected = tuple(Section(e["file"], e["heading"]) for e in data["expected"])
        if not expected:
            raise ValueError(f"{path}:{line_no}: question has no expected sections")
        questions.append(Question(data["id"], data["type"], data["question"], expected))

    duplicates = [i for i, n in Counter(q.id for q in questions).items() if n > 1]
    if duplicates:
        raise ValueError(f"{path}: duplicate question ids {duplicates}")
    return questions


def is_match(result: SearchResult, section: Section) -> bool:
    """True if the chunk is in the expected section or one of its subsections."""
    return (
        result.doc_path == section.file
        and section.heading in result.heading_path.split(" > ")
    )


def first_hit_rank(
    results: list[SearchResult], expected: tuple[Section, ...]
) -> int | None:
    """Rank (from 1) of the first result in any expected section; None if there is none."""
    for rank, result in enumerate(results, start=1):
        if any(is_match(result, section) for section in expected):
            return rank
    return None


def hit_rate(ranks: list[int | None], k: int) -> float:
    """Share of questions whose first correct chunk is in the top k."""
    return sum(r is not None and r <= k for r in ranks) / len(ranks)


def mean_reciprocal_rank(ranks: list[int | None], k: int) -> float:
    """Average of 1/rank; a question with no correct chunk in the top k counts as 0."""
    return sum(1 / r for r in ranks if r is not None and r <= k) / len(ranks)


def missing_sections(conn, questions: list[Question]) -> list[tuple[str, Section]]:
    """Expected sections that no chunk in the database belongs to (typos, renamed headings)."""
    rows = conn.execute("SELECT DISTINCT doc_path, heading_path FROM chunks").fetchall()
    known = {(doc, title) for doc, path in rows for title in path.split(" > ")}
    return [
        (q.id, s)
        for q in questions
        for s in q.expected
        if (s.file, s.heading) not in known
    ]


class _PrecomputedEmbedder:
    """Hands out embeddings computed in one batch up front.

    Each question is embedded once, and hybrid_search still runs unchanged
    (it calls embedder.embed itself).
    """

    def __init__(self, texts: list[str], vectors: list[list[float]]):
        self._vectors = dict(zip(texts, vectors))

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vectors[t] for t in texts]


def run_eval(conn, embedder, questions: list[Question]) -> dict[str, list[int | None]]:
    """Rank of the first correct chunk for every question, per mode."""
    texts = [q.question for q in questions]
    precomputed = _PrecomputedEmbedder(texts, embedder.embed(texts))

    ranks: dict[str, list[int | None]] = {mode: [] for mode in MODES}
    for q in questions:
        [query_vector] = precomputed.embed([q.question])
        results = {
            "vector": search.vector_search(conn, query_vector, k=DEPTH),
            "keyword": search.keyword_search(conn, q.question, k=DEPTH),
            "hybrid": search.hybrid_search(conn, precomputed, q.question, k=DEPTH),
        }
        for mode in MODES:
            ranks[mode].append(first_hit_rank(results[mode], q.expected))
    return ranks


def format_report(
    questions: list[Question], ranks: dict[str, list[int | None]], k: int = 5
) -> str:
    """Per-question ranks, then hit@1, hit@k and MRR@k per mode, then hit@k per question type."""
    lines = [
        f"Rank of the first correct chunk in each mode's top {DEPTH} (- = not found).",
        f"The metrics below count only the top {k}.",
        "",
        f"{'id':<5}{'type':<12}{'vec':>4}{'kw':>4}{'hyb':>5}  question",
    ]
    for i, q in enumerate(questions):
        shown = ["-" if ranks[m][i] is None else str(ranks[m][i]) for m in MODES]
        text = q.question if len(q.question) <= 70 else q.question[:69] + "…"
        lines.append(
            f"{q.id:<5}{q.type:<12}{shown[0]:>4}{shown[1]:>4}{shown[2]:>5}  {text}"
        )

    lines += [
        "",
        f"{len(questions)} questions",
        f"{'mode':<9}{'hit@1':>7}{f'hit@{k}':>7}{f'MRR@{k}':>8}",
    ]
    for mode in MODES:
        r = ranks[mode]
        lines.append(
            f"{mode:<9}{hit_rate(r, 1):>7.0%}{hit_rate(r, k):>7.0%}{mean_reciprocal_rank(r, k):>8.2f}"
        )

    lines += [
        "",
        f"hit@{k} by question type",
        f"{'type':<12}{'n':>3}" + "".join(f"{m:>9}" for m in MODES),
    ]
    for qtype in sorted({q.type for q in questions}):
        idx = [i for i, q in enumerate(questions) if q.type == qtype]
        cells = "".join(
            f"{hit_rate([ranks[m][i] for i in idx], k):>9.0%}" for m in MODES
        )
        lines.append(f"{qtype:<12}{len(idx):>3}{cells}")
    return "\n".join(lines)
