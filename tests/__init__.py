"""Test-suite guard: no test may make a real network call.

Every paid API this project touches — Gemini image, Veo video, and now Gemini
text for the `frontier` profile — is reached over the network, so "the tests
are offline" is a COST guarantee, not just a speed one. It had been true by
convention (every test mocks its transport) but nothing enforced it: one
missing `mock.patch` in a new test would quietly start billing, and the
symptom would be a slightly slower suite, not a failure.

This makes that guarantee mechanical. Outbound socket connections raise
`NetworkBlockedInTests` with a pointed message. Correctly-mocked tests never
reach the socket layer, so they are unaffected; a test that forgot to mock
fails loudly and immediately, naming the address it tried to reach.

Deliberately blocking at the SOCKET layer rather than patching `urlopen` or
`gemini._post`: those are per-module, so every new client module would need
its own guard, and an SDK doing its own HTTP (`google-genai`, say) would slip
straight past. Every network path in every library ends at `socket.connect`.

Loopback is blocked too, which is intended: a test must not depend on a live
Ollama daemon either, since that makes results differ between a machine
that happens to be serving one and CI. Local IPC that never calls `connect`
with an address (`socketpair`, pre-connected fds) is unaffected.
"""
from __future__ import annotations

import socket


class NetworkBlockedInTests(RuntimeError):
    """Raised when a test attempts a real outbound connection."""


_MESSAGE = (
    "Blocked a real network connection to {addr} from the test suite.\n"
    "Tests must never call a live API — Gemini/Veo calls cost real money, and "
    "a test that reaches Ollama depends on whether a daemon happens to be "
    "running.\n"
    "Mock the transport instead: patch `gemini._post` (or `gemini.generate_text`"
    " / `gemini.available`) for hosted calls, and `llm.generate` or "
    "`llm.urllib.request.urlopen` for local ones. See tests/__init__.py."
)

_real_socket = socket.socket
_real_create_connection = socket.create_connection


class _GuardedSocket(_real_socket):
    def connect(self, address, *args, **kwargs):
        raise NetworkBlockedInTests(_MESSAGE.format(addr=address))

    def connect_ex(self, address, *args, **kwargs):
        raise NetworkBlockedInTests(_MESSAGE.format(addr=address))


def _blocked_create_connection(address, *args, **kwargs):
    raise NetworkBlockedInTests(_MESSAGE.format(addr=address))


socket.socket = _GuardedSocket
socket.create_connection = _blocked_create_connection
