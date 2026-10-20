"""Ingest: docs -> chunks -> embeddings -> database."""

from pgvector import Vector

from rag.chunking import Chunk, chunk_markdown
from rag.config import Settings
from rag.db import connect, init_db
from rag.embed import get_embedder
from rag.markdown_utils import estimate_tokens
from rag.sources import ensure_docs, github_url, select_files


def build_chunks(settings: Settings) -> list[Chunk]:
    docs_dir = ensure_docs(settings)
    chunks: list[Chunk] = []
    for doc_path in select_files(docs_dir):
        text = (docs_dir / doc_path).read_text(encoding="utf-8")
        chunks.extend(chunk_markdown(text, doc_path))
    return chunks


def ingest(settings: Settings) -> None:
    chunks = build_chunks(settings)
    texts = [c.embedding_text for c in chunks]
    total_tokens = sum(estimate_tokens(t) for t in texts)
    print(
        f"{len(chunks)} chunks, ~{total_tokens:,} tokens to embed "
        f"(provider: {settings.embedding_provider})"
    )

    # Embed everything BEFORE touching the database, so a failed API call
    # leaves the existing data untouched.
    vectors = get_embedder(settings).embed(texts)

    rows = [
        (
            c.doc_path,
            c.heading_path,
            c.part,
            github_url(settings, c.doc_path, c.anchor),
            c.content,
            estimate_tokens(c.embedding_text),
            Vector(vector),
            settings.docs_commit,
        )
        for c, vector in zip(chunks, vectors, strict=True)
    ]

    with connect(settings.database_url) as conn:
        init_db(conn, settings.embedding_dims)
        # Replace all rows in one transaction: anyone searching sees either the
        # old data or the new data, never a half-loaded table.
        with conn.transaction():
            conn.execute("DELETE FROM chunks")
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO chunks (doc_path, heading_path, part, url, content,
                                        token_count, embedding, source_commit)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    rows,
                )
    print(f"Stored {len(rows)} chunks.")
