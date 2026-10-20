"""Search: vector, keyword (full-text) and hybrid."""

from dataclasses import dataclass

from pgvector import Vector


@dataclass(frozen=True)
class SearchResult:
    chunk_id: int
    doc_path: str
    heading_path: str
    url: str
    content: str
    score: float


def vector_search(
    conn, query_embedding: list[float], k: int = 20
) -> list[SearchResult]:
    """The k chunks whose embeddings are closest to the query's (cosine distance)."""
    sql = """
        SELECT id, doc_path, heading_path, url, content,
            1 - (%s <=> embedding) AS similarity
        FROM chunks
        ORDER BY %s <=> embedding
        LIMIT %s;
    """
    vec = Vector(query_embedding)
    rows = conn.execute(sql, (vec, vec, k)).fetchall()
    return [SearchResult(*row) for row in rows]


def keyword_search(conn, query: str, k: int = 20) -> list[SearchResult]:
    """The k best full-text matches for the query, ranked by ts_rank."""
    raise NotImplementedError


def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]], k: int = 60
) -> list[SearchResult]:
    """Merge ranked lists: each chunk scores the sum of 1 / (k + rank) over the lists."""
    raise NotImplementedError


def hybrid_search(conn, embedder, query: str, k: int = 5) -> list[SearchResult]:
    """Vector search + keyword search, merged with reciprocal_rank_fusion."""
    raise NotImplementedError
