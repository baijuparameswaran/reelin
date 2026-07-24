#!/usr/bin/env python3
"""Concatenate two or more video files into one, in the given order.

Uses ffmpeg's concat demuxer: tries a fast, lossless stream-copy first
(works when all inputs share the same codec/resolution/framerate), and
falls back to a re-encode (H.264/AAC) if that fails — e.g. when the inputs
don't match closely enough for stream-copy concat.

Usage:
    python scripts/stitch_videos.py vid1.mp4 vid2.mp4 vid3.mp4 -o movie.mp4
    python scripts/stitch_videos.py vid1.mp4 vid2.mp4 vid3.mp4   # writes
        # movie.mp4 next to the first input if -o is omitted

Requires ffmpeg on PATH.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def stitch(clips: list[Path], out_path: Path, *, reencode: bool = False) -> bool:
    """Concatenate `clips` (in the given order) into `out_path`. Returns
    True on success. Tries stream-copy first (fast, lossless); falls back
    to a re-encode if the inputs don't match closely enough for that."""
    missing = [p for p in clips if not p.exists()]
    if missing:
        print(f"error: file(s) not found: {', '.join(str(p) for p in missing)}", file=sys.stderr)
        return False
    if shutil.which("ffmpeg") is None:
        print("error: ffmpeg not found on PATH", file=sys.stderr)
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for p in clips:
            f.write(f"file '{p.resolve()}'\n")
        list_file = Path(f.name)

    base = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file)]
    attempts = ([base + ["-c", "copy", "-movflags", "+faststart", str(out_path)]]
                if not reencode else [])
    attempts.append(base + ["-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-c:a", "aac", "-movflags", "+faststart", str(out_path)])
    try:
        for i, cmd in enumerate(attempts):
            r = subprocess.run(cmd, capture_output=True, timeout=600)
            if r.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
                how = "stream-copy" if (not reencode and i == 0) else "re-encode"
                print(f"stitched {len(clips)} clip(s) → {out_path} ({how})")
                return True
        print(f"stitch failed: {r.stderr.decode(errors='replace')[-500:].strip()}", file=sys.stderr)
        return False
    finally:
        list_file.unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="+", help="video files to concatenate, in order")
    ap.add_argument("-o", "--out", default=None,
                    help="output path (default: movie.mp4 next to the first input)")
    ap.add_argument("--reencode", action="store_true",
                    help="skip the stream-copy attempt and re-encode directly")
    args = ap.parse_args()

    clips = [Path(c) for c in args.clips]
    out_path = Path(args.out) if args.out else clips[0].parent / "movie.mp4"

    ok = stitch(clips, out_path, reencode=args.reencode)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
