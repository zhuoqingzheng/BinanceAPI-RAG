"""Get the Binance docs at the pinned commit and choose which files to index."""

import subprocess
from pathlib import Path

from rag.config import Settings


def ensure_docs(settings: Settings) -> Path:
    """Download the docs repo at settings.docs_commit, if it isn't there already.

    Fetches only that one commit (--depth 1), so it's a few MB.
    """
    docs_dir = settings.docs_dir
    if _current_commit(docs_dir) == settings.docs_commit:
        return docs_dir

    docs_dir.mkdir(parents=True, exist_ok=True)
    if not (docs_dir / ".git").exists():
        _git(docs_dir, "init", "-q")
        _git(docs_dir, "remote", "add", "origin", settings.docs_repo_url)
    _git(docs_dir, "fetch", "-q", "--depth", "1", "origin", settings.docs_commit)
    _git(docs_dir, "checkout", "-q", "--force", "FETCH_HEAD")
    return docs_dir


def select_files(docs_dir: Path) -> list[str]:
    """Paths (relative to the repo root) of the English docs we index.

    Skipped: Chinese translations (*_CN.md), testnet/ copies (near-duplicates of
    the main docs), changelogs and terms of use.
    """
    candidates = [
        *docs_dir.glob("*.md"),
        *docs_dir.glob("faqs/*.md"),
        *docs_dir.glob("demo-mode/*.md"),
    ]
    selected = []
    for path in candidates:
        name = path.name
        if name.endswith("_CN.md") or "TERMS-OF-USE" in name or "CHANGELOG" in name:
            continue
        selected.append(path.relative_to(docs_dir).as_posix())
    return sorted(selected)


def github_url(settings: Settings, doc_path: str, anchor: str) -> str:
    """Link to a section on GitHub, pinned to the same commit we indexed."""
    base = settings.docs_repo_url.removesuffix(".git")
    return f"{base}/blob/{settings.docs_commit}/{doc_path}#{anchor}"


def _current_commit(docs_dir: Path) -> str | None:
    if not (docs_dir / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(docs_dir), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(cwd), *args], check=True)
