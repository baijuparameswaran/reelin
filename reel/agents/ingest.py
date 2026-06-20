"""Ingest agent: load and normalize raw source material.

Accepts any plain-text book / short story / script and returns a normalized
record the downstream creative agents can rely on. (Richer formats — PDF, EPUB,
.fountain, .fdx — are a later iteration; this keeps the thin slice robust.)
"""
from __future__ import annotations

import re
from pathlib import Path


def ingest(path: str | Path) -> dict:
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="replace")

    # Normalize line endings and collapse excessive blank lines.
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # Naive title guess: a short first non-empty line reads like a title.
    first_line = next((l.strip() for l in text.splitlines() if l.strip()), "")
    title = first_line if 0 < len(first_line) <= 80 else p.stem.replace("_", " ").title()

    return {
        "source_path": str(p),
        "title": title,
        "text": text,
        "word_count": len(text.split()),
        "char_count": len(text),
    }
