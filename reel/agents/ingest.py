"""Ingest agent: load and normalize raw source material.

Accepts plain-text or PDF books / short stories / scripts and returns a
normalized record the downstream creative agents can rely on. The text is also
split into overlapping chunks so that per-scene agents can reference only the
portion of the story relevant to their scene rather than truncating to a fixed
head.

Supported formats:
  .txt / .md / .fountain / any plain-text — read directly.
  .pdf                                    — text extracted via pypdf (install
                                            with: pip install pypdf).
"""
from __future__ import annotations

import re
from pathlib import Path

# Chunk tuning.  3 000-char chunks with 300-char overlap give ~1 chunk per scene
# for a typical short story and keep each chunk well inside any model's context.
CHUNK_SIZE = 3000
CHUNK_OVERLAP = 300


def chunk_text(text: str,
               size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """Split text into overlapping chunks, preferring paragraph/sentence breaks.

    Returns a list of {"index", "start", "end", "text"} records. Adjacent
    chunks overlap by `overlap` characters so that scenes that straddle a
    boundary appear fully in at least one chunk.
    """
    chunks: list[dict] = []
    start = 0
    idx = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + size, text_len)
        # Try to break at a natural boundary within the trailing 20% of the chunk.
        if end < text_len:
            search_from = max(start, end - max(200, size // 5))
            boundary = text.rfind("\n\n", search_from, end)
            if boundary == -1:
                boundary = text.rfind(". ", search_from, end)
            if boundary != -1:
                end = boundary + 1       # include the punctuation / blank line
        chunks.append({"index": idx, "start": start, "end": end,
                        "text": text[start:end]})
        if end >= text_len:
            break                        # reached the end — no more chunks
        next_start = end - overlap
        if next_start <= start:          # guard against degenerate short text
            next_start = start + 1
        start = next_start
        idx += 1
    return chunks


def scene_source_context(source: dict,
                          chunk_indices: list[int] | None,
                          max_chars: int = 6000) -> str:
    """Return the source text relevant to a scene from its chunk indices.

    Falls back to the head of the full text when chunks are absent (old
    checkpoints) or no indices are mapped yet. Adjacent chunks are joined
    with a separator; overlap at boundaries is accepted (the model ignores it).
    """
    chunks: list[dict] = source.get("chunks") or []
    full_text: str = source.get("text", "")

    if not chunks or not chunk_indices:
        # No chunk data — return the head of the full text.
        return full_text[:max_chars]

    # Collect chunks in index order, deduplicated.
    seen: set[int] = set()
    parts: list[str] = []
    for idx in sorted(set(chunk_indices)):
        if idx in seen or idx < 0 or idx >= len(chunks):
            continue
        seen.add(idx)
        parts.append(chunks[idx]["text"])

    if not parts:
        return full_text[:max_chars]

    combined = "\n".join(parts)
    if len(combined) > max_chars:
        combined = combined[:max_chars] + "\n[… excerpt truncated]"
    return combined


def _extract_pdf(p: Path) -> str:
    """Extract plain text from a PDF using pypdf.

    Handles searchable PDFs (the typical case for digital screenplays and
    scripts).  Scanned-image PDFs will return empty/garbled text — those need
    an OCR pass before ingestion.
    """
    try:
        import pypdf
    except ImportError as exc:
        raise ImportError(
            "pypdf is required for PDF ingestion.  Install it with:\n"
            "    pip install pypdf"
        ) from exc

    reader = pypdf.PdfReader(p)
    pages: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text(extraction_mode="layout") or ""
        # Strip lone page-number lines common in screenplays (e.g. "42." or "42")
        page_text = re.sub(r"(?m)^[ \t]*\d+\.?[ \t]*$", "", page_text)
        pages.append(page_text)

    text = "\n\n".join(pages)

    # Collapse runs of spaces introduced by PDF layout positioning, but only
    # within a line so screenplay indentation is preserved.
    text = re.sub(r"[ \t]{3,}", "  ", text)
    return text


def ingest(path: str | Path) -> dict:
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        text = _extract_pdf(p)
    else:
        text = p.read_text(encoding="utf-8", errors="replace")

    # Normalize line endings and collapse excessive blank lines.
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # Naive title guess: a short first non-empty line reads like a title.
    first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    title = (first_line if 0 < len(first_line) <= 80
             else p.stem.replace("_", " ").title())

    chunks = chunk_text(text)

    return {
        "source_path": str(p),
        "title": title,
        "text": text,
        "word_count": len(text.split()),
        "char_count": len(text),
        "chunks": chunks,
        "chunk_count": len(chunks),
    }
