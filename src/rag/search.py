"""Search: vector, keyword (full-text) and hybrid."""

from dataclasses import dataclass, replace

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
    """The k best full-text matches for the query, ranked by ts_rank.

    A chunk matches if it contains any of the query's words, not all of them
    """
    # plainto_tsquery drops stopwords, stems the rest and ANDs them: '-1021' & 'mean'.
    # Swapping ' & ' for ' | ' makes it OR. The query is built once, in FROM, as q.
    # `id` breaks ties so equal ranks always come back in the same order.
    sql = """
        SELECT id, doc_path, heading_path, url, content, ts_rank(tsv, q) AS rank
        FROM chunks,
             CAST(replace(plainto_tsquery('english', %s)::text, ' & ', ' | ') AS tsquery) AS q
        WHERE tsv @@ q
        ORDER BY rank DESC, id
        LIMIT %s
    """
    rows = conn.execute(sql, (query, k)).fetchall()
    return [SearchResult(*row) for row in rows]


def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]], k: int = 60
) -> list[SearchResult]:
    """Merge ranked lists: each chunk scores the sum of 1 / (k + rank) over the lists."""
    scoreLookUp: dict[int, float] = {}
    resultLookUp: dict[int, SearchResult] = {}
    response = []
    for result_list in result_lists:
        for rank, result in enumerate(result_list, start = 1):
            chunk_id = result.chunk_id
            if (resultLookUp.get(chunk_id) is None):
                scoreLookUp[chunk_id] = 0
                resultLookUp[chunk_id] = result
            scoreLookUp[chunk_id] += 1 / (k + rank) 
            resultLookUp[chunk_id] = result

    ranked_ids = sorted(scoreLookUp, key=lambda cid: scoreLookUp[cid], reverse=True)
    for id in ranked_ids:
        result = resultLookUp[id]
        result = replace(result, score=scoreLookUp[id])
        response.append(result)
    return response
    



def hybrid_search(conn, embedder, query: str, k: int = 5) -> list[SearchResult]:
    """Vector search + keyword search, merged with reciprocal_rank_fusion."""
    
