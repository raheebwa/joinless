# SPDX-License-Identifier: MIT
"""Shared test configuration.

**Hypothesis's per-example time limit is off, deliberately.** Its default fails an example
that takes longer than 200 ms, which is a timing assertion wearing a correctness test's
clothes. The
property tests in ``tests/test_corpus.py`` generate a full 1800-pair corpus per example:
warm, that costs single-digit milliseconds, but on a machine doing anything else at the
same time it has been measured at 250–640 ms. The result is a suite that goes red because
of what else the machine was doing, and Hypothesis says so itself when it happens —
*"Failed on the first call but did not on a subsequent one"*.

A false red is worse than a missing signal here, because it trains a reader to re-run
rather than to look. And the signal is not lost: what those tests assert is that different
seeds produce different content and that every pair lands in exactly one role — properties
with no timing semantics at all. Generation cost *is* measured, deliberately and in
isolation, by the benchmark harness (RFC-0002), which reports it with warm-up counts,
repetition counts and percentiles rather than a single wall-clock sample taken while a
test runner competes for the CPU.
"""

from __future__ import annotations

from hypothesis import HealthCheck, settings

settings.register_profile(
    "joinless",
    # Hypothesis names this setting for the wall-clock limit it enforces; `None` removes
    # that limit. See this module's docstring for why a limit is wrong here.
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("joinless")
