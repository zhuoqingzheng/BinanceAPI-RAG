# Binance API RAG

A question-answering assistant over [Binance's Spot API documentation](https://github.com/binance/binance-spot-api-docs). It finds the relevant doc sections with **hybrid search** (vector search with pgvector plus Postgres full-text search, merged with Reciprocal Rank Fusion) and answers with **citations** that link to the exact section on GitHub.

Stack: Python 3.12, PostgreSQL 17 + pgvector (Docker), OpenAI embeddings and chat models, uv, pytest.

> Status: the full pipeline works: `rag ask` retrieves with hybrid search and answers with citations. Next: a retrieval eval (hit@5 and MRR per search mode) to measure vector vs keyword vs hybrid.

## Example

```
$ uv run rag ask "what does -1021 mean"
-1021 INVALID_TIMESTAMP — "Timestamp for this request is outside of the recvWindow."
Possible cause: "Timestamp for this request was 1000ms ahead of the server's time." [1]

Sources:
[1] errors.md > Error codes for Binance > 10xx - General Server or Network issues > -1021 INVALID_TIMESTAMP
    https://github.com/binance/binance-spot-api-docs/blob/f0ee79c.../errors.md#-1021-invalid_timestamp
```

Exact strings like error codes are where hybrid search helps: vector search alone ranks the neighbouring codes -1020 and -1022 almost as high as -1021, while keyword search matches `-1021` exactly.

## How it works

```
INGEST (once)
  docs repo @ pinned commit → English Markdown files
    → chunk by heading (strip TOC, keep code blocks whole, split long sections)
    → embed each chunk (OpenAI text-embedding-3-small)
    → Postgres: text + vector + full-text index

QUERY
  question ─┬→ embed → vector search (top 20) ─────┐
            └→ full-text keyword search (top 20) ──┴→ merge (RRF) → top 5 chunks
            → LLM answers from those chunks only, citing them

EVAL (next)
  questions with known correct sections → hit@k and MRR, per search mode
```
<img width="1014" height="784" alt="image" src="https://github.com/user-attachments/assets/1bbd6bea-5e8c-4f0f-9a02-8ce9c1b9a825" />

### Retrieval and answers

| Step | How it works |
|---|---|
| **Vector search** | The question is embedded with the same model as the chunks. The 20 chunks with the smallest cosine distance (pgvector's `<=>`) are returned. |
| **Keyword search** | The question goes through `plainto_tsquery('english', …)`, which drops stopwords and stems the rest, and its words are joined with OR. Chunks are matched against the stored `tsv` column (GIN index) and ranked with `ts_rank`, with heading words weighted above body words. Top 20. |
| **Fusion** | Reciprocal Rank Fusion: each chunk scores the sum of `1 / (60 + rank)` over the two lists, so a chunk found by both searches beats one found by only one. The top 5 go to the answer step. |
| **Answer** | The LLM (`ANSWER_MODEL`, default `gpt-5-mini`) gets the 5 chunks numbered `[1]`–`[5]`, with instructions to answer only from them and cite them by number. The CLI then prints a link for each cited section. |

## Setup

Requirements: Docker, [uv](https://docs.astral.sh/uv/), git.

```bash
cp .env.example .env          # then add your OPENAI_API_KEY
docker compose up -d          # Postgres + pgvector on localhost:5433
uv sync                       # install dependencies
uv run rag fetch-docs         # download the docs at the pinned commit
uv run rag init-db            # create the table and indexes
uv run rag ingest             # chunk, embed and store the docs (about US$0.005)
uv run rag ask "How do I place a new order?"
```

## Commands

| Command | What it does | API cost |
|---|---|---|
| `uv run rag fetch-docs` | Download the docs at the pinned commit | none |
| `uv run rag chunks --stats` | Chunk all docs and print statistics | none |
| `uv run rag chunks --file filters.md` | Print the chunks for one file | none |
| `uv run rag ingest` | Chunk, embed and store everything (replaces existing rows) | ~244k tokens ≈ US$0.005 at $0.02 per 1M tokens (Oct 2026) |
| `uv run rag init-db --reset` | Drop and recreate the table, deleting all chunks. Needed after changing `EMBEDDING_DIMS` | none |
| `uv run rag search "your question"` | Search the docs and list the top sections (`--mode` hybrid, vector or keyword; `-k` results) | one tiny embedding call (hybrid and vector modes) |
| `uv run rag ask "your question"` | Answer from the top 5 chunks (`--mode` and `-k` as for `search`), citing them, with links to each cited section | one embedding call + one `ANSWER_MODEL` call |
| `uv run pytest` | Run the tests: chunking rules and the answer step, with a fake LLM | none |

Set `EMBEDDING_PROVIDER=fake` in `.env` to run `ingest` offline with meaningless vectors, which is useful for testing the pipeline without spending anything.

## Design decisions

- **English docs only, pinned to one commit**, so chunk counts, citations and eval results are reproducible. Citation links point at that commit too, so they keep working when the docs change. Chinese translations, testnet copies, changelogs and terms of use are excluded.
- **Chunk by heading.** Each section of the docs is one topic. Every chunk carries its heading path (e.g. `Filters > Symbol filters > LOT_SIZE`), which is embedded along with the text. Sections over ~800 tokens are split at blank lines outside code blocks, so code examples and tables stay whole. The 25 files give 708 chunks.
- **One Postgres for both searches.** At ~700 chunks, exact vector search takes milliseconds, so there's no vector index (an approximate index like HNSW could miss results), and no separate search engine.
- **Headings weighted higher in full-text search**, since error codes and filter names usually appear in headings.
- **Keyword search matches any word, not every word.** `plainto_tsquery` requires all of them, so "what does -1021 mean" found nothing: the -1021 section never uses the word "mean". With OR it's found, and ranking orders the matches.
- **Fuse by rank, not by score.** Cosine similarity and `ts_rank` are on different scales (`ts_rank` levels off near 1.0 when words repeat), so adding the scores would favour whichever runs larger.
- **Safe re-ingest.** Ingest embeds everything before touching the database, then replaces all rows in one transaction, so a failed API call or a search during ingest never sees a half-loaded table.
- **Swappable LLM.** The answer step talks to a small `LLM` interface, so another provider can replace OpenAI without changing the rest. The tests use a fake LLM and cost nothing.

## Known limitations

- **`ts_rank` has no IDF.** A rare term like `-1021` counts no more than a common word like "mean", so keyword search alone ranks FAQs titled "What does that field mean?" above the -1021 section. Hybrid search recovers here because vector search ranks -1021 first. A BM25 extension (such as ParadeDB's `pg_search`) would fix it at the source.
- **Parts of one long section compete for the top 5.** For "How do I place a new order?", 3 of the 5 hybrid results are parts of the WebSocket "Place new order" section, and the REST endpoint `POST /api/v3/order` drops out, so the answer only covers WebSocket.
- **Token counts are estimated** (characters / 4), not counted with a tokenizer.

## Project layout

```
src/rag/
  cli.py             the `rag` command
  config.py          settings from .env; the pinned docs commit
  sources.py         fetch the docs, select the English files, build GitHub URLs
  chunking.py        Markdown -> chunks (rules numbered to match the tests)
  markdown_utils.py  token estimate and GitHub heading anchors
  embed.py           OpenAI embedder (batches of 100, retries) and a fake one for offline runs
  db.py              schema: chunks table, full-text column and GIN index
  ingest.py          chunk -> embed -> replace all rows in one transaction
  search.py          vector, keyword and hybrid search; Reciprocal Rank Fusion
  answer.py          prompt, LLM interface, citation parsing
tests/               chunking rules; answer step with a fake LLM
eval/                retrieval questions (in progress)
```
