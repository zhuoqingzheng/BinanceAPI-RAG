"""Settings, read from environment variables (and a .env file if present)."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The docs are pinned to one commit so that chunk counts, citations and eval
# results are reproducible. Bump this deliberately, then re-run ingest and eval.
DOCS_REPO_URL = "https://github.com/binance/binance-spot-api-docs.git"
DOCS_COMMIT = "f0ee79c8ce24dbeda6645211b9504e237fddef19"  # 2026-10-15


@dataclass(frozen=True)
class Settings:
    database_url: str
    openai_api_key: str | None
    embedding_provider: str  # "openai" or "fake"
    embedding_model: str
    embedding_dims: int
    answer_model: str
    docs_dir: Path
    docs_repo_url: str = DOCS_REPO_URL
    docs_commit: str = DOCS_COMMIT


def load_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")
    return Settings(
        database_url=os.getenv("DATABASE_URL", "postgresql://rag:rag@localhost:5433/rag"),
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "openai"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        embedding_dims=int(os.getenv("EMBEDDING_DIMS", "1536")),
        answer_model=os.getenv("ANSWER_MODEL", "gpt-5-mini"),
        docs_dir=PROJECT_ROOT / "data" / "binance-spot-api-docs",
    )
