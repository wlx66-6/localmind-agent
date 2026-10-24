from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt"}
SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".idea",
    ".vscode",
}
MAX_FILE_BYTES = 5 * 1024 * 1024


def iter_source_files(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS or part.startswith(".") for part in path.relative_to(root).parts):
            continue
        if path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
