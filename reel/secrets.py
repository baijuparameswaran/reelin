"""Gemini API key management for reel.

Stores the key in a permissions-protected file at ~/.config/reel/gemini_key
(chmod 600, readable only by the current user). The key is never written to
.bashrc, environment variables, or anywhere in the project directory.

Why a file rather than keyrings.cryptfile: the cryptfile backend prompts for its
own encryption password via getpass on every read, which raises EOFError in any
non-interactive context (piped output, VSCode integrated terminal, --resume runs,
background batch jobs) — causing silent render skips. A chmod-600 file gives the
same effective isolation without requiring interactive TTY access.

Usage:
    python -m reel.secrets set       # store the key (prompts securely)
    python -m reel.secrets get       # confirm a key is stored (prints masked)
    python -m reel.secrets delete    # remove the stored key
    python -m reel.secrets status    # show key status
"""
from __future__ import annotations

import stat
import sys
from pathlib import Path

from .gemini import _KEY_FILE, _invalidate_key_cache


def _ensure_dir() -> None:
    _KEY_FILE.parent.mkdir(parents=True, exist_ok=True)


def _mask(key: str) -> str:
    if len(key) <= 8:
        return "****"
    return key[:4] + "****" + key[-4:]


def cmd_set() -> None:
    import getpass
    print("Enter your Gemini API key (input hidden):")
    try:
        key = getpass.getpass("GEMINIAPIKEY: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        sys.exit(1)
    if not key:
        print("Aborted — no key entered.")
        sys.exit(1)
    _ensure_dir()
    _KEY_FILE.write_text(key + "\n", encoding="utf-8")
    _KEY_FILE.chmod(stat.S_IRUSR | stat.S_IWUSR)   # 0o600 — owner read/write only
    _invalidate_key_cache()
    print(f"Key stored at {_KEY_FILE} ({_mask(key)}, permissions 600).")


def cmd_get() -> None:
    if not _KEY_FILE.exists():
        print("No key stored — run `python -m reel.secrets set`.")
        return
    try:
        v = _KEY_FILE.read_text(encoding="utf-8").strip()
        if v:
            print(f"Key found at {_KEY_FILE} ({_mask(v)})")
        else:
            print(f"Key file exists but is empty ({_KEY_FILE}).")
    except Exception as e:
        print(f"Key file unreadable ({_KEY_FILE}): {e}")


def cmd_delete() -> None:
    if _KEY_FILE.exists():
        _KEY_FILE.unlink()
        _invalidate_key_cache()
        print(f"Key deleted ({_KEY_FILE}).")
    else:
        print("No key stored — nothing to delete.")


def cmd_status() -> None:
    if not _KEY_FILE.exists():
        print(f"  key file : ✗ not found ({_KEY_FILE})")
        print("  Run `python -m reel.secrets set` to store your Gemini API key.")
        print("  image/video stages will be skipped.")
        return
    try:
        v = _KEY_FILE.read_text(encoding="utf-8").strip()
        perms = oct(_KEY_FILE.stat().st_mode)[-3:]
        if v:
            print(f"  key file : ✓ key stored ({_mask(v)}, permissions {perms})")
            if perms != "600":
                print(f"  ⚠ permissions are {perms}, not 600 — run: chmod 600 {_KEY_FILE}")
            print("  image/video stages will use Gemini.")
        else:
            print(f"  key file : ✗ file exists but is empty ({_KEY_FILE})")
            print("  Run `python -m reel.secrets set` to store your Gemini API key.")
    except Exception as e:
        print(f"  key file : ✗ unreadable ({_KEY_FILE}): {e}")


def main() -> None:
    cmds = {"set": cmd_set, "get": cmd_get, "delete": cmd_delete, "status": cmd_status}
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd not in cmds:
        print(f"Usage: python -m reel.secrets [{' | '.join(cmds)}]")
        sys.exit(2)
    cmds[cmd]()


if __name__ == "__main__":
    main()
