"""Meta-test: the suite cannot spend money.

`tests/__init__.py` blocks outbound sockets so a test that forgets to mock its
transport fails loudly instead of quietly billing a Gemini/Veo call. That
guard is itself untested code in the most literal sense — if someone deletes
it, or a refactor stops it from being imported, every other test still passes
and the protection is simply gone, with no signal. These tests fail in that
case.

Run with: python -m unittest tests.test_no_api_cost -v
"""
from __future__ import annotations

import socket
import unittest
from pathlib import Path

import tests as _tests_pkg
from reel import gemini, llm


class TestNetworkIsBlocked(unittest.TestCase):
    def test_the_guard_is_installed(self):
        self.assertIs(socket.socket, _tests_pkg._GuardedSocket)
        self.assertIs(socket.create_connection, _tests_pkg._blocked_create_connection)

    def test_a_raw_outbound_connection_raises(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            with self.assertRaises(_tests_pkg.NetworkBlockedInTests):
                s.connect(("example.com", 80))

    def test_connect_ex_is_blocked_too(self):
        """`connect_ex` returns an errno rather than raising, so a caller
        using it would otherwise slip past a `connect`-only guard."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            with self.assertRaises(_tests_pkg.NetworkBlockedInTests):
                s.connect_ex(("example.com", 80))

    def test_create_connection_is_blocked(self):
        with self.assertRaises(_tests_pkg.NetworkBlockedInTests):
            socket.create_connection(("example.com", 80))

    def test_loopback_is_blocked_so_a_live_ollama_cannot_change_results(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            with self.assertRaises(_tests_pkg.NetworkBlockedInTests):
                s.connect(("127.0.0.1", 11434))


class TestPaidPathsCannotFire(unittest.TestCase):
    """The three real-money entry points, exercised unmocked. Each must fail
    at the socket rather than reaching the API."""

    def _assert_blocked(self, fn, *a, **kw):
        with self.assertRaises(Exception) as cm:
            fn(*a, **kw)
        chain, err = [], cm.exception
        while err is not None:
            chain.append(err)
            err = err.__cause__ or err.__context__
        self.assertTrue(
            any(isinstance(e, _tests_pkg.NetworkBlockedInTests) for e in chain),
            f"expected a blocked connection, got {cm.exception!r}")

    def test_gemini_text_is_blocked(self):
        self._assert_blocked(gemini.generate_text, "hi", model="gemini-3.6-flash")

    def test_gemini_image_is_blocked(self):
        self._assert_blocked(gemini.generate_image, "a cat", Path("/tmp/nope.png"))

    def test_local_generate_never_reaches_a_daemon(self):
        """The Ollama path degrades before the socket — `installed_models()`
        swallows the blocked call and `resolve_model` then reports no models —
        so assert on the outcome rather than the guard exception: either way
        no daemon was contacted."""
        with self.assertRaises(RuntimeError) as cm:
            llm.generate("hi", profile="fast")
        self.assertIn("No Ollama models are installed", str(cm.exception))


class TestSuiteIsCredentialIndependent(unittest.TestCase):
    """Results must not differ between a machine with a Gemini key and one
    without — otherwise the suite passes locally and behaves differently in
    CI, and a cost-relevant branch (hosted vs local routing) goes untested on
    whichever side you don't happen to be on."""

    def test_max_chars_for_frontier_is_pinned_by_mocking_not_by_the_real_key(self):
        from unittest import mock
        with mock.patch.object(gemini, "available", return_value=True):
            hosted = llm.max_chars("frontier")
        with mock.patch.object(gemini, "available", return_value=False):
            degraded = llm.max_chars("frontier")
        self.assertGreater(hosted, degraded)
        # Both branches are asserted above regardless of whether this machine
        # actually has a key — that is the point.
