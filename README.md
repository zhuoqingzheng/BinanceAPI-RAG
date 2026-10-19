# Binance API RAG

A question-answering assistant over [Binance's Spot API documentation](https://github.com/binance/binance-spot-api-docs). It retrieves the relevant doc sections with **hybrid search** (vector search with pgvector + Postgres full-text search, merged with Reciprocal Rank Fusion) and answers with **citations** linking to the exact section on GitHub. A small **eval set** measures how often retrieval finds the right section.

> Status: work in progress. Ingest pipeline and database done; chunking, search, answers and eval in progress.

## How it works

```
INGEST (once)
  docs repo @ pinned commit → English Markdown files
    → chunk by heading (strip TOC, keep code blocks whole, split long sections)
    → embed each chunk (OpenAI text-embedding-3-small)
    → Postgres: text + vector + full-text index

QUERY
  question ─┬→ embed → vector search ──────┐
            └→ full-text keyword search ───┴→ merge (RRF) → top chunks
            → LLM answers from those chunks only, citing them

EVAL
  questions with known correct sections → hit@k and MRR, per search mode
```
<img width="1014" height="784" alt="image" src="https://github.com/user-attachments/assets/1bbd6bea-5e8c-4f0f-9a02-8ce9c1b9a825" />


## Setup

Requirements: Docker, [uv](https://docs.astral.sh/uv/), git.

```bash
cp .env.example .env          # then add your OPENAI_API_KEY
docker compose up -d          # Postgres + pgvector on localhost:5433
uv sync                       # install dependencies
uv run rag fetch-docs         # download the docs at the pinned commit
uv run rag init-db            # create the table and indexes
```

## Commands

| Command | What it does | API cost |
|---|---|---|
| `uv run rag fetch-docs` | Download the docs at the pinned commit | none |
| `uv run rag chunks --stats` | Chunk all docs and print statistics | none |
| `uv run rag chunks --file filters.md` | Print the chunks for one file | none |
| `uv run rag ingest` | Chunk, embed and store everything (replaces existing rows) | ~244k tokens ≈ US$0.005 at $0.02 per 1M tokens (Oct 2026) |
| `uv run rag init-db --reset` | Drop and recreate the table, deleting all chunks. Needed after changing `EMBEDDING_DIMS` | none |
| `uv run rag search "your question"` | Search the docs (`--mode` hybrid, vector or keyword; `-k` results). Not implemented yet | one tiny embedding call (hybrid and vector modes) |
| `uv run rag ask "your question"` | Answer from the top 5 chunks, citing them, with links to each cited section | one embedding call + one `ANSWER_MODEL` call |
| `uv run pytest` | Run the tests | none |

Set `EMBEDDING_PROVIDER=fake` in `.env` to run `ingest` offline with meaningless vectors, which is useful for testing the pipeline without spending anything.

## Design decisions

- **English docs only, pinned to one commit**, so chunk counts, citations and eval results are reproducible. Chinese translations, testnet copies, changelogs and terms of use are excluded.
- **Chunk by heading.** Each section of the docs is one topic. Every chunk carries its heading path (e.g. `Filters > Symbol filters > LOT_SIZE`), which is embedded along with the text.
- **One Postgres for both searches.** At ~700 chunks, exact vector search takes milliseconds, so there's no vector index (an approximate index like HNSW could miss results), and no separate search engine.
- **Headings weighted higher in full-text search**, since error codes and filter names usually appear in headings.
