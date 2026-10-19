"""Command-line entry point: `uv run rag <command>`."""

import argparse
from collections import Counter

from rag.config import load_settings
from rag.markdown_utils import estimate_tokens


def cmd_fetch_docs(args, settings):
    from rag.sources import ensure_docs, select_files

    docs_dir = ensure_docs(settings)
    print(f"Docs at {docs_dir} (commit {settings.docs_commit[:7]}), "
          f"{len(select_files(docs_dir))} files selected.")


def cmd_init_db(args, settings):
    from rag.db import connect, init_db

    with connect(settings.database_url) as conn:
        if args.reset:
            conn.execute("DROP TABLE IF EXISTS chunks")
            print("Dropped table chunks.")
        init_db(conn, settings.embedding_dims)
    print("Database ready.")


def cmd_chunks(args, settings):
    from rag.ingest import build_chunks
    from rag.sources import github_url

    chunks = build_chunks(settings)
    if args.file:
        chunks = [c for c in chunks if c.doc_path == args.file]
        if not chunks:
            raise SystemExit(f"No chunks for {args.file!r}. Check the name with: uv run rag chunks --stats")

    if args.stats:
        sizes = [estimate_tokens(c.embedding_text) for c in chunks]
        per_file = Counter(c.doc_path for c in chunks)
        print(f"{len(per_file)} files, {len(chunks)} chunks, ~{sum(sizes):,} tokens")
        print(f"tokens per chunk: median {sorted(sizes)[len(sizes) // 2]}, max {max(sizes)}")
        print(f"chunks over 800 tokens: {sum(s > 800 for s in sizes)}")
        print(f"sections split into parts: {len({(c.doc_path, c.heading_path) for c in chunks if c.part > 1})}")
        print("\nchunks per file:")
        for doc_path, n in sorted(per_file.items()):
            print(f"  {n:4}  {doc_path}")
        return

    for i, c in enumerate(chunks[: args.limit], start=1):
        content = c.content if args.full else _shorten(c.content, 300)
        print(f"── [{i}] {c.heading_path}  (part {c.part}, ~{estimate_tokens(c.embedding_text)} tokens)")
        print(f"   {github_url(settings, c.doc_path, c.anchor)}")
        print(content)
        print()


def cmd_ingest(args, settings):
    from rag.ingest import ingest

    ingest(settings)


def _search(conn, settings, query: str, mode: str, k: int):
    from rag import search
    from rag.embed import get_embedder

    if mode == "keyword":
        return search.keyword_search(conn, query, k=k)
    if mode == "vector":
        [query_vector] = get_embedder(settings).embed([query])
        return search.vector_search(conn, query_vector, k=k)
    return search.hybrid_search(conn, get_embedder(settings), query, k=k)


def cmd_search(args, settings):
    from rag.db import connect

    with connect(settings.database_url) as conn:
        results = _search(conn, settings, args.query, args.mode, args.k)
    for rank, r in enumerate(results, start=1):
        print(f"{rank}. [{r.score:.4f}] {r.doc_path} > {r.heading_path}\n   {r.url}")


def cmd_ask(args, settings):
    from rag.answer import answer_question, cited_sources, get_llm
    from rag.db import connect

    llm = get_llm(settings)  # fails fast if there's no API key
    with connect(settings.database_url) as conn:
        results = _search(conn, settings, args.question, args.mode, args.k)
    answer = answer_question(args.question, results, llm)
    print(answer.text)
    # Show the sources the answer cites; all of them if it cites none.
    sources = cited_sources(answer) or list(enumerate(answer.sources, start=1))
    if sources:
        print("\nSources:")
    for number, r in sources:
        print(f"[{number}] {r.doc_path} > {r.heading_path}\n    {r.url}")


def _shorten(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + " …"


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag", description="RAG over Binance's Spot API docs")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("fetch-docs", help="download the docs at the pinned commit").set_defaults(func=cmd_fetch_docs)

    p = sub.add_parser("init-db", help="create the table and indexes")
    p.add_argument("--reset", action="store_true", help="drop the table first (deletes all chunks)")
    p.set_defaults(func=cmd_init_db)

    p = sub.add_parser("chunks", help="preview chunking (no API calls)")
    p.add_argument("--file", help="only this doc, e.g. filters.md")
    p.add_argument("--stats", action="store_true", help="print statistics instead of chunks")
    p.add_argument("--limit", type=int, default=20, help="max chunks to print (default 20)")
    p.add_argument("--full", action="store_true", help="print full chunk text")
    p.set_defaults(func=cmd_chunks)

    sub.add_parser("ingest", help="chunk, embed and store all docs").set_defaults(func=cmd_ingest)

    p = sub.add_parser("search", help="search the docs")
    p.add_argument("query")
    p.add_argument("--mode", choices=["hybrid", "vector", "keyword"], default="hybrid")
    p.add_argument("-k", type=int, default=5)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("ask", help="answer a question from the docs, with citations")
    p.add_argument("question")
    # Vector for now; switch the default to hybrid once hybrid_search is written.
    p.add_argument("--mode", choices=["hybrid", "vector", "keyword"], default="vector")
    p.add_argument("-k", type=int, default=5, help="number of chunks the answer is based on")
    p.set_defaults(func=cmd_ask)

    args = parser.parse_args()
    try:
        args.func(args, load_settings())
    except NotImplementedError as e:
        raise SystemExit(f"Not implemented yet: {e}") from None


if __name__ == "__main__":
    main()
