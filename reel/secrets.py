"""Encrypted API key management for reel.

Stores the Gemini API key in the system keyring (keyrings.cryptfile on WSL2 —
an AES-encrypted file at ~/.local/share/python_keyring/). The key is never
written to .bashrc, env files, or anywhere in the project directory.

Usage:
    python -m reel.secrets set       # store the key (prompts securely)
    python -m reel.secrets get       # confirm a key is stored (prints masked)
    python -m reel.secrets delete    # remove the stored key
    python -m reel.secrets status    # show where the key comes from
"""
from __future__ import annotations

import sys

from .gemini import _KEYRING_SERVICE, _KEYRING_USERNAME


def _require_keyring():
    try:
        import keyring
        return keyring
    except ImportError:
        print("keyring not installed — run: pip install keyring keyrings.cryptfile")
        sys.exit(1)


def _mask(key: str) -> str:
    if len(key) <= 8:
        return "****"
    return key[:4] + "****" + key[-4:]


def cmd_set() -> None:
    import getpass
    kr = _require_keyring()
    print("Enter your Gemini API key (input hidden):")
    key = getpass.getpass("GEMINIAPIKEY: ").strip()
    if not key:
        print("Aborted — no key entered.")
        sys.exit(1)
    kr.set_password(_KEYRING_SERVICE, _KEYRING_USERNAME, key)
    print(f"Key stored in keyring ({_mask(key)}). "
          f"You can now remove it from .bashrc.")


def cmd_get() -> None:
    kr = _require_keyring()
    v = kr.get_password(_KEYRING_SERVICE, _KEYRING_USERNAME)
    if v:
        print(f"Keyring: key found ({_mask(v)})")
    else:
        print("Keyring: no key stored.")


def cmd_delete() -> None:
    kr = _require_keyring()
    try:
        kr.delete_password(_KEYRING_SERVICE, _KEYRING_USERNAME)
        print("Key deleted from keyring.")
    except Exception as e:
        print(f"Nothing to delete (or error: {e})")


def cmd_status() -> None:
    kr = _require_keyring()
    v = kr.get_password(_KEYRING_SERVICE, _KEYRING_USERNAME)
    if v:
        print(f"  keyring : ✓ key stored ({_mask(v)})")
        print("  image/video stages will use Gemini.")
    else:
        print("  keyring : ✗ no key stored — run `python -m reel.secrets set`")
        print("  image/video stages will be skipped.")


def main() -> None:
    cmds = {"set": cmd_set, "get": cmd_get, "delete": cmd_delete, "status": cmd_status}
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd not in cmds:
        print(f"Usage: python -m reel.secrets [{' | '.join(cmds)}]")
        sys.exit(2)
    cmds[cmd]()


if __name__ == "__main__":
    main()
