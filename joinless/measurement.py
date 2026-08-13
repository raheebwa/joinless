# SPDX-License-Identifier: MIT
"""Resource measurement, isolated per arm per metric (RFC-0002 Method step 7).

Every metric this module reports — warm scoring latency, peak resident memory,
and the cold-start phases — shares one mechanism and one failure shape.

**One mechanism.** Two arms measured in one process contaminate each other: a
warm allocator, a page cache, or an already-imported runtime left behind by
whichever arm ran first makes the next arm's figure describe the wrong thing
(RFC-0002 Method step 7). ``_run_in_child`` is the single place this module
spawns a fresh ``sys.executable`` interpreter — never a fork, because the whole
point is a process that has imported nothing yet — and every public function
below calls it exactly once. A second, independent way to spawn a worker would
be the isolation failure this module exists to prevent, arrived at from
inside its own file.

**One failure shape.** ADR-0013 requires an arm that cannot initialise to be
recorded as unavailable with a reason rather than omitted, dropped, or scored
as zero. That has to hold whether the failure is the arm's own — a scorer
whose dependency is not installed, or one that does not exist yet, such as the
milestone-M3 embedding arms — or the worker's — a crash, a timeout's absence
of output, anything that stops a result reaching stdout. ``_measure`` is the
one place a worker is spawned and its failure turned into :class:`Unavailable`,
so every public function shares that mapping instead of re-deriving it.

Reported figures follow the same rule for a different reason: RFC-0002's
Metrics table says a classical arm's cold start has no session-creation or
tokenizer-load phase at all, not that those phases take zero time. That is
the same "undefined is not zero" fact ADR-0013 states for precision and
recall, so this module imports :class:`joinless.evaluation.Metric` rather than
defining a second type for the same distinction.

Each worker script below is a self-contained string, not an importable
function. That is deliberate for the cold-start phases in particular: timing
``import joinless.scoring`` requires that nothing in the child process have
imported it yet, which an invocation of the shape ``from joinless.measurement
import worker; worker()`` could not give — the parent package would already be
imported by the time such a call is reachable. A plain string handed to
``sys.executable -c`` carries no such cost, and it is the same mechanism
``tests/test_import_boundary.py`` already uses for the same reason.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from joinless import resolver
from joinless.evaluation import Metric
from joinless.records import Record
from joinless.scoring import Scorer


@dataclass(frozen=True, slots=True)
class ArtifactRequirement:
    """A model artefact an arm needs before it can be measured (ADR-0013's
    fourth fail-closed rule, issue #51). Neither a missing file nor a checksum
    mismatch is repaired by fetching a replacement — see :func:`verify_artifact`,
    which reports why rather than attempting one. Fetching a substitute would
    let a run proceed on an artefact other than the one its record names,
    which is exactly the confound ADR-0013 forbids.
    """

    path: Path
    sha256: str


class _ChildFailed(RuntimeError):
    """A worker exited non-zero, or produced no parseable JSON. Both are a
    worker failure (issue #53's third bullet) rather than a crash of the
    parent — :func:`_measure` turns this into :class:`Unavailable` instead of
    letting it propagate."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


def _run_in_child(script: str, params: Mapping[str, Any]) -> dict[str, Any]:
    """Run ``script`` in a fresh ``sys.executable`` interpreter — never a
    fork, so the process has imported nothing yet — passing ``params`` through
    an environment variable rather than string-formatting them into the
    script, so no value in ``params`` can be interpreted as code.

    This is the single isolation mechanism every metric in this module uses
    (see the module docstring). ``script`` reads its parameters from
    ``JOINLESS_MEASURE_PARAMS`` and prints exactly one line of JSON before
    exiting; anything else — a non-zero exit, or stdout that does not parse —
    is a worker failure and raises :class:`_ChildFailed` rather than being
    interpreted as a result.
    """
    env = {**os.environ, "JOINLESS_MEASURE_PARAMS": json.dumps(params)}
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if result.returncode != 0:
        raise _ChildFailed(
            result.stderr.strip() or f"worker exited with status {result.returncode}"
        )
    try:
        return cast(dict[str, Any], json.loads(result.stdout))
    except ValueError as exc:
        raise _ChildFailed(f"worker produced no parseable output: {exc}") from exc


def verify_artifact(requirement: ArtifactRequirement) -> str | None:
    """``None`` when ``requirement.path`` exists and hashes to
    ``requirement.sha256``; otherwise the reason it does not.

    Never attempts to fetch a replacement for a missing or mismatched file —
    refusing is the correct response (ADR-0013), not a fallback with a
    convenient repair.
    """
    if not requirement.path.is_file():
        return f"artifact missing at {requirement.path}"
    digest = hashlib.sha256(requirement.path.read_bytes()).hexdigest()
    if digest != requirement.sha256:
        return (
            f"artifact checksum mismatch at {requirement.path}: "
            f"expected {requirement.sha256}, got {digest}"
        )
    return None


@dataclass(frozen=True, slots=True)
class Unavailable:
    """An arm that could not be measured: it failed to initialise, its worker
    crashed, or its artefact was missing or did not match (ADR-0013, issue #51,
    issue #53). Every measurement function below returns this instead of a
    partial or zeroed result, so the arm keeps its row in the table rather
    than being dropped from it.
    """

    arm: str
    reason: str


def _measure(
    arm: str,
    artifact: ArtifactRequirement | None,
    script: str,
    params: Mapping[str, Any],
) -> dict[str, Any] | Unavailable:
    """The one place every metric checks its artefact and spawns its worker.

    A metric-specific function only builds ``script`` and ``params`` and turns
    a successful payload into its own result type; this is what stops there
    being two ways in this module to spawn a worker or to decide an arm is
    unavailable (issue #53).
    """
    if artifact is not None:
        reason = verify_artifact(artifact)
        if reason is not None:
            return Unavailable(arm=arm, reason=reason)
    try:
        payload = _run_in_child(script, params)
    except _ChildFailed as exc:
        return Unavailable(arm=arm, reason=exc.reason)
    if payload.get("status") == "unavailable":
        return Unavailable(arm=arm, reason=cast(str, payload["reason"]))
    return payload


def _percentile(data: Sequence[float], fraction: float) -> float:
    """The one percentile convention every repeated-sample metric in this
    module shares (RFC-0002 Method step 3: "repeat and report the
    distribution, never a single run"). A module-level function rather than
    logic re-typed inline in each worker script, so ``_PREPARATION_COST_SCRIPT``
    (issue #103) computes p50/p99 exactly the way ``_WARM_LATENCY_SCRIPT``
    already does, both importing this one implementation, instead of the
    formula existing twice and risking the two drifting apart.

    Clamped at the last index rather than indexing past it — ``int(fraction *
    len(data))`` reaches ``len(data)`` exactly when ``fraction`` is ``1.0``,
    which would be an out-of-range index into ``ordered`` were it not
    clamped.
    """
    ordered = sorted(data)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


# Reads its parameters from JOINLESS_MEASURE_PARAMS and prints exactly one
# line of JSON (see _run_in_child). get_scorer's two documented failure modes
# — an unknown arm name and a known arm with a missing dependency (ADR-0013,
# joinless.scoring) — are both "cannot initialise" here; nothing else is
# caught, so a genuine bug in the worker still crashes it and is reported by
# _run_in_child as a worker failure rather than masked as an arm failure.
_WARM_LATENCY_SCRIPT = """
import json
import os
import time

import joinless.scoring as scoring
from joinless.measurement import _percentile

params = json.loads(os.environ["JOINLESS_MEASURE_PARAMS"])
try:
    scorer = scoring.get_scorer(params["arm"])
except (ValueError, scoring.ScorerUnavailable) as exc:
    print(json.dumps({"status": "unavailable", "reason": str(exc)}))
else:
    left = scorer.prepare(params["left_name"])
    right = scorer.prepare(params["right_name"])
    for _ in range(params["warmup_count"]):
        scorer.score(left, right)

    durations = []
    for _ in range(params["repetition_count"]):
        _start = time.perf_counter()
        scorer.score(left, right)
        durations.append(time.perf_counter() - _start)

    print(json.dumps({
        "status": "ok",
        "p50_seconds": _percentile(durations, 0.50),
        "p99_seconds": _percentile(durations, 0.99),
    }))
"""


# What the warm-latency figure covers, carried on every value it produces. The
# timed section is `score` alone: both operands are prepared once before it, so
# the graph a neural arm runs never executes inside it. Stated on the figure
# because the alternative is a reader concluding from two near-identical warm
# p50s that quantization bought nothing, when the inference cost it does move
# sits in preparation cost instead (issue #65, ADR-0009).
WARM_LATENCY_SCOPE = (
    "score only; both operands are prepared before the timed section, so no "
    "preparation cost is included in this figure"
)


@dataclass(frozen=True, slots=True)
class WarmLatency:
    """Per-comparison scoring latency at the median and the 99th percentile,
    after warm-up (RFC-0002 Metrics, issue #54). ``warmup_count`` and
    ``repetition_count`` travel with the figures rather than living only in
    whatever call produced them, so a reader holding this value alone can
    tell how it was measured."""

    arm: str
    p50_seconds: float
    p99_seconds: float
    warmup_count: int
    repetition_count: int
    scope: str = WARM_LATENCY_SCOPE


def measure_warm_latency(
    arm: str,
    left_name: str,
    right_name: str,
    *,
    warmup_count: int,
    repetition_count: int,
    artifact: ArtifactRequirement | None = None,
) -> WarmLatency | Unavailable:
    """Warm-scoring p50/p99 for one comparison, in a fresh isolated worker
    (RFC-0002 Method steps 2 and 7, issue #54).

    ``repetition_count`` must be at least 1 and ``warmup_count`` must not be
    negative — both are validated before anything is spawned, so a caller
    error is a ``ValueError`` here rather than a confusing shape from the
    worker.
    """
    if repetition_count < 1:
        raise ValueError("repetition_count must be at least 1")
    if warmup_count < 0:
        raise ValueError("warmup_count must not be negative")

    params = {
        "arm": arm,
        "left_name": left_name,
        "right_name": right_name,
        "warmup_count": warmup_count,
        "repetition_count": repetition_count,
    }
    result = _measure(arm, artifact, _WARM_LATENCY_SCRIPT, params)
    if isinstance(result, Unavailable):
        return result
    return WarmLatency(
        arm=arm,
        p50_seconds=cast(float, result["p50_seconds"]),
        p99_seconds=cast(float, result["p99_seconds"]),
        warmup_count=warmup_count,
        repetition_count=repetition_count,
    )


def _preparation_costs(
    scorer: Scorer[Any],
    left_names: Sequence[str],
    right_names: Sequence[str],
    left_indices: Sequence[int],
    right_indices: Sequence[int],
    *,
    warmup_count: int,
    repetition_count: int,
) -> tuple[list[float], list[float]]:
    """Time both preparation paths for an already-constructed ``scorer``,
    over :class:`~joinless.records.Record` objects rebuilt from
    ``left_names``/``right_names`` and the ``(left_indices, right_indices)``
    pairs they form — the same shape ``_PREPARATION_COST_SCRIPT`` decodes
    from ``JOINLESS_MEASURE_PARAMS`` and passes straight through.

    Both timed sections call :func:`joinless.resolver.prepare_hoisted` and
    :func:`joinless.resolver.prepare_naive` — the exact functions
    :func:`joinless.resolver.score_candidates` calls to do the real scoring
    the library ships — rather than a second copy of either loop living in
    this module (issue #100). ``joinless.resolver`` is imported as a module
    (not ``from ... import prepare_hoisted``) so a test can monkeypatch
    ``resolver.prepare_hoisted``/``resolver.prepare_naive`` and observe this
    function actually call the patched version, proving delegation rather
    than a coincidentally-matching copy.

    Importable and callable directly, outside the isolated worker, which is
    what makes that delegation testable in-process without paying for a
    subprocess spawn; ``_PREPARATION_COST_SCRIPT`` calls this same function,
    for real, inside a fresh interpreter, purely to get realistic wall-clock
    figures (RFC-0002 Method step 7).

    A single call of either path is one draw, not a measurement (issue
    #103: seven direct samples of a smaller workload gave ratios from 0.89
    to 1.14, with the hoisted path occasionally measuring *slower* than its
    control — noise, reported as if it were signal). Each path is therefore
    run ``warmup_count`` times, discarded, then ``repetition_count`` times,
    timed — the same discard-then-collect discipline
    :data:`_WARM_LATENCY_SCRIPT` already applies to ``score``, applied here
    to ``prepare_hoisted``/``prepare_naive`` instead of a second convention
    for "repeat and report the distribution" (RFC-0002 Method step 3). The
    two paths get their own warm-up rather than sharing one: a batched
    ``prepare_all`` call and a loop of per-comparison ``prepare`` calls are
    different access patterns, and only running each pattern itself brings
    it to steady state. Nothing here computes a percentile — this function
    returns the raw per-repetition durations for each path, and
    ``_PREPARATION_COST_SCRIPT`` reduces them with the same
    :func:`_percentile` :data:`_WARM_LATENCY_SCRIPT` reduces its own with,
    so both metrics use one summarising function, not two.
    """
    left_records = [
        Record(source="preparation-cost", ordinal=index, name=name)
        for index, name in enumerate(left_names)
    ]
    right_records = [
        Record(source="preparation-cost", ordinal=index, name=name)
        for index, name in enumerate(right_names)
    ]
    pairs = [
        (left_records[left_index], right_records[right_index])
        for left_index, right_index in zip(left_indices, right_indices, strict=True)
    ]

    for _ in range(warmup_count):
        resolver.prepare_hoisted(scorer, left_records, right_records)
    hoisted_durations = []
    for _ in range(repetition_count):
        start = time.perf_counter()
        resolver.prepare_hoisted(scorer, left_records, right_records)
        hoisted_durations.append(time.perf_counter() - start)

    for _ in range(warmup_count):
        resolver.prepare_naive(scorer, pairs)
    naive_durations = []
    for _ in range(repetition_count):
        start = time.perf_counter()
        resolver.prepare_naive(scorer, pairs)
        naive_durations.append(time.perf_counter() - start)

    return hoisted_durations, naive_durations


_PREPARATION_COST_SCRIPT = """
import json
import os

import joinless.scoring as scoring
from joinless.measurement import _percentile, _preparation_costs

params = json.loads(os.environ["JOINLESS_MEASURE_PARAMS"])
try:
    scorer = scoring.get_scorer(params["arm"])
except (ValueError, scoring.ScorerUnavailable) as exc:
    print(json.dumps({"status": "unavailable", "reason": str(exc)}))
else:
    hoisted_durations, naive_durations = _preparation_costs(
        scorer,
        params["left_names"],
        params["right_names"],
        params["left_indices"],
        params["right_indices"],
        warmup_count=params["warmup_count"],
        repetition_count=params["repetition_count"],
    )
    print(json.dumps({
        "status": "ok",
        "hoisted_p50_seconds": _percentile(hoisted_durations, 0.50),
        "hoisted_p99_seconds": _percentile(hoisted_durations, 0.99),
        "naive_p50_seconds": _percentile(naive_durations, 0.50),
        "naive_p99_seconds": _percentile(naive_durations, 0.99),
    }))
"""


@dataclass(frozen=True, slots=True)
class PreparationCost:
    """Hoisted and naive preparation cost for one arm, over the same
    candidate set whose bucket occupancy the run record carries alongside it
    (issue #66) — never quoted without the occupancy that produced it
    (ADR-0009: "a hoist speed-up quoted without the occupancy that produced
    it cannot be transferred to any other dataset").

    Reported at the median and the 99th percentile, over repeated
    warmed-up samples of each path — the same p50/p99 shape
    :class:`WarmLatency` already establishes, not a second convention
    (issue #103). A single draw of either path is not a measurement:
    preparation cost for a small workload is itself small, so scheduling
    noise is a large fraction of it, and seven direct samples at a smaller
    workload gave hoist ratios from 0.89 to 1.14 — at the low end the
    hoisted path measured *slower* than its naive control. ``..._p50_seconds``
    is the typical cost a reader comparing arms should compare; ``..._p99_seconds``
    is the tail, exactly as RFC-0002's Metrics table gives the same reason
    for warm scoring's own p99 ("what a user feels").

    ``hoisted_p50_seconds``/``hoisted_p99_seconds`` time ``prepare_all``
    called once per side, batched — the production pattern ADR-0009
    requires. ``naive_p50_seconds``/``naive_p99_seconds`` time ``prepare``
    called fresh for both records of every comparison in
    ``comparison_count``, reproducing the redundant recomputation the hoist
    removes. Both are timed in the same isolated worker (RFC-0002 Method
    step 7 names "preparation cost" alongside cold start, warm scoring and
    peak RSS as a metric requiring isolation).

    ``warmup_count`` and ``repetition_count`` travel with the figures for
    the same reason :class:`WarmLatency` carries its own: a reader holding
    one ``PreparationCost`` alone can tell how it was made, not only what it
    says. ``record_count`` and ``comparison_count`` travel with them too, so
    a reader can also see what dataset produced it without cross-referencing
    another part of the run record.
    """

    arm: str
    hoisted_p50_seconds: float
    hoisted_p99_seconds: float
    naive_p50_seconds: float
    naive_p99_seconds: float
    warmup_count: int
    repetition_count: int
    record_count: int
    comparison_count: int


def measure_preparation_cost(
    arm: str,
    left_names: Sequence[str],
    right_names: Sequence[str],
    comparison_pairs: Sequence[tuple[int, int]],
    *,
    warmup_count: int,
    repetition_count: int,
    artifact: ArtifactRequirement | None = None,
) -> PreparationCost | Unavailable:
    """Hoisted vs naive preparation cost for ``arm`` over
    ``comparison_pairs`` — each pair a ``(left_index, right_index)`` into
    ``left_names``/``right_names`` — in a fresh isolated worker (RFC-0002
    Method step 7, issue #66), sampled repeatedly after warm-up the same way
    :func:`measure_warm_latency` already samples ``score`` (issue #103).

    ``left_names``, ``right_names`` and ``comparison_pairs`` must each be
    non-empty, ``repetition_count`` must be at least 1 and ``warmup_count``
    must not be negative — all validated before anything is spawned, so a
    caller error is a ``ValueError`` here rather than a confusing shape from
    the worker.
    """
    if not left_names:
        raise ValueError("left_names must not be empty")
    if not right_names:
        raise ValueError("right_names must not be empty")
    if not comparison_pairs:
        raise ValueError("comparison_pairs must not be empty")
    if repetition_count < 1:
        raise ValueError("repetition_count must be at least 1")
    if warmup_count < 0:
        raise ValueError("warmup_count must not be negative")

    params = {
        "arm": arm,
        "left_names": list(left_names),
        "right_names": list(right_names),
        "left_indices": [pair[0] for pair in comparison_pairs],
        "right_indices": [pair[1] for pair in comparison_pairs],
        "warmup_count": warmup_count,
        "repetition_count": repetition_count,
    }
    result = _measure(arm, artifact, _PREPARATION_COST_SCRIPT, params)
    if isinstance(result, Unavailable):
        return result
    return PreparationCost(
        arm=arm,
        hoisted_p50_seconds=cast(float, result["hoisted_p50_seconds"]),
        hoisted_p99_seconds=cast(float, result["hoisted_p99_seconds"]),
        naive_p50_seconds=cast(float, result["naive_p50_seconds"]),
        naive_p99_seconds=cast(float, result["naive_p99_seconds"]),
        warmup_count=warmup_count,
        repetition_count=repetition_count,
        record_count=len(left_names) + len(right_names),
        comparison_count=len(comparison_pairs),
    )


# resource.getrusage's ru_maxrss unit is platform-dependent — kilobytes on
# Linux, bytes everywhere this project supports otherwise (ADR-0002 scopes out
# Windows) — so the worker normalises to bytes itself rather than leaving the
# unit to whichever platform happened to run the measurement.
_PEAK_MEMORY_SCRIPT = """
import json
import os
import resource
import sys
import threading

import joinless.scoring as scoring

params = json.loads(os.environ["JOINLESS_MEASURE_PARAMS"])
try:
    scorer = scoring.get_scorer(params["arm"])
except (ValueError, scoring.ScorerUnavailable) as exc:
    print(json.dumps({"status": "unavailable", "reason": str(exc)}))
else:
    left = scorer.prepare(params["left_name"])
    right = scorer.prepare(params["right_name"])
    scorer.score(left, right)

    _raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    _peak_bytes = _raw * 1024 if sys.platform.startswith("linux") else _raw

    print(json.dumps({
        "status": "ok",
        "peak_rss_bytes": _peak_bytes,
        "thread_count": threading.active_count(),
    }))
"""


@dataclass(frozen=True, slots=True)
class PeakMemory:
    """Peak resident set size for one arm's whole lifetime, including model
    load, measured in its own process (RFC-0002 Metrics, issue #55).

    The sample is taken once, after ``scorer.prepare`` and ``scorer.score``
    have both run, immediately before the worker exits — so it covers
    everything the worker did, by construction, for whichever arm runs,
    model load included for the two embedding arms that now exist
    (:mod:`joinless.embedding`). Exercising that claim in a test needs a
    real model artefact on disk, which this suite does not carry (issue
    #59), so "including model load" holds by construction — the sampling
    point comes after everything the arm's worker does — rather than from a
    test that has measured a model-loading arm's peak RSS.

    ``thread_count`` and ``power_mode`` travel with the figure because both
    move it (issue #55's third bullet): ``power_mode`` is supplied by the
    caller rather than detected here — reading a platform's power state is a
    platform-specific concern no issue in this batch asks this module to
    build, so the field exists to carry a value the caller already has,
    not to derive one.

    ``thread_count`` is ``threading.active_count()`` (:data:`_PEAK_MEMORY_SCRIPT`)
    — a count of Python-level ``threading.Thread`` objects alive in the
    worker at sampling time, and nothing else. It cannot see a native thread
    pool a C extension creates on its own, such as ONNX Runtime's intra-op
    pool for a neural arm's session: that pool's threads are never
    ``threading.Thread`` instances, so this field reads the same for a
    classical arm and a neural arm regardless of how many native threads the
    latter's runtime actually used (issue #109). What this project
    configured for that pool (nothing) and what ONNX Runtime therefore used
    (its own automatic default, unpinned) are two further, distinct facts —
    recorded on :class:`~joinless.runrecord.Environment`, not here, since
    they describe the runtime's own thread pool rather than a count this
    field could ever have reported.
    """

    arm: str
    peak_rss_bytes: int
    thread_count: int
    power_mode: str


def measure_peak_memory(
    arm: str,
    left_name: str,
    right_name: str,
    *,
    power_mode: str,
    artifact: ArtifactRequirement | None = None,
) -> PeakMemory | Unavailable:
    """Peak RSS for ``arm``'s whole lifetime, in a fresh isolated worker
    (RFC-0002 Method step 7, issue #55)."""
    params = {"arm": arm, "left_name": left_name, "right_name": right_name}
    result = _measure(arm, artifact, _PEAK_MEMORY_SCRIPT, params)
    if isinstance(result, Unavailable):
        return result
    return PeakMemory(
        arm=arm,
        peak_rss_bytes=cast(int, result["peak_rss_bytes"]),
        thread_count=cast(int, result["thread_count"]),
        power_mode=power_mode,
    )


# Timing "import" here requires that nothing in this process have imported
# joinless.scoring yet — see the module docstring for why that rules out an
# invocation of the shape "from joinless.measurement import worker; worker()"
# for this one metric specifically: by the time such a call is reachable, the
# parent package is already imported. The script below imports it itself,
# after starting its own clock, and is the only worker in this module that
# has to.
#
# scoring.get_scorer(...) is timed too (issue #108): previously it sat between
# two clock reads and its cost was measured by nobody. For a neural arm, that
# one call is where ONNX Runtime and the tokenizer package are imported
# (EmbeddingScorer's own construction path, joinless.embedding._build_scorer)
# and where the session and tokenizer are actually built from the artefact —
# exactly the costs RFC-0002's Metrics table calls "import" (the arm's own
# imports), "session creation" and "tokenizer load". A classical arm's scorer
# carries neither of the latter two (getattr(..., None)), so the whole
# construction duration is folded into import — which is correct for
# `overlap` and `fuzzy` too: whatever a classical arm's own constructor
# imports (rapidfuzz, for `fuzzy`) belongs under "the arm's own imports", not
# under a phase named for a session or tokenizer neither one has.
_COLD_START_SCRIPT = """
import json
import os
import sys
import time

_child_start = time.time()
params = json.loads(os.environ["JOINLESS_MEASURE_PARAMS"])

_before_import = time.perf_counter()
import joinless.scoring as scoring
_after_import = time.perf_counter()

_before_construct = time.perf_counter()
try:
    scorer = scoring.get_scorer(params["arm"])
except (ValueError, scoring.ScorerUnavailable) as exc:
    print(json.dumps({"status": "unavailable", "reason": str(exc)}))
    sys.exit(0)
_after_construct = time.perf_counter()

_tokenizer_load_seconds = getattr(scorer, "tokenizer_load_seconds", None)
_session_creation_seconds = getattr(scorer, "session_creation_seconds", None)
_construction_seconds = _after_construct - _before_construct
if _tokenizer_load_seconds is None or _session_creation_seconds is None:
    _import_remainder_seconds = _construction_seconds
else:
    _import_remainder_seconds = (
        _construction_seconds - _tokenizer_load_seconds - _session_creation_seconds
    )

_before_first = time.perf_counter()
left = scorer.prepare(params["left_name"])
right = scorer.prepare(params["right_name"])
scorer.score(left, right)
_after_first = time.perf_counter()

print(json.dumps({
    "status": "ok",
    "child_start_epoch": _child_start,
    "import_seconds": (_after_import - _before_import) + _import_remainder_seconds,
    "session_creation_seconds": _session_creation_seconds,
    "tokenizer_load_seconds": _tokenizer_load_seconds,
    "first_inference_seconds": _after_first - _before_first,
}))
"""

# RFC-0002 Metrics table, "Cold start — session creation" / "— tokenizer load":
# null for the classical arms, which construct no session / load no tokenizer.
# Pinned as literals rather than derived from the RFC text at runtime: a reader
# meets these strings in a run record, so they are part of what this module
# reports and a test can assert their content. Text generated from a document at
# runtime would change silently when the document did, which is the opposite of
# what a record needs.
_NO_SESSION_REASON = "classical arms construct no session"
_NO_TOKENIZER_REASON = "classical arms load no tokenizer"


def _phase_or_reason(seconds: float | None, reason: str) -> Metric:
    """A cold-start sub-phase that only some arms have: defined when the
    worker reported a real duration, undefined with ``reason`` when it did
    not — the same branch :func:`measure_artifact_size` already makes on
    whether ``paths`` came back empty (issue #63), generalised here from "no
    artefact paths" to "no duration reported" (issue #108). The worker is
    the one place that knows which arm actually constructed a session or a
    tokenizer (:data:`_COLD_START_SCRIPT`'s ``getattr(scorer, ..., None)``),
    so this function branches on what it reported rather than re-deriving
    "is this arm neural" from the arm's name a second time.
    """
    if seconds is None:
        return Metric(value=None, undefined_reason=reason)
    return Metric(value=seconds, undefined_reason=None)


# RFC-0002 Method step 7 / Metrics table: interpreter start is identical
# across every arm and is not attributable to any of them. Every other phase
# is. A frozenset of phase names, attached to each ColdStartPhases instance
# below, is what lets a reader holding one value alone answer "which phases
# are this arm's own cost" without re-deriving the rule from the RFC.
_NOT_ATTRIBUTABLE_PHASES: frozenset[str] = frozenset({"interpreter start"})


def _sum_defined(phases: Sequence[Metric]) -> Metric:
    """Cold start (total): the sum of whichever phases produced a duration,
    skipping — never zeroing — any phase that is undefined. This is the same
    rule :func:`joinless.evaluation._aggregate` applies when pooling counts
    across a family with an undefined metric (ADR-0013 point 1): a total that
    coerced a classical arm's null session-creation phase to zero would be
    fabricating a duration for a step that did not happen, not reporting one
    that took no time.
    """
    defined = [phase.value for phase in phases if phase.value is not None]
    if not defined:
        return Metric(value=None, undefined_reason="no phase produced a duration")
    return Metric(value=sum(defined), undefined_reason=None)


@dataclass(frozen=True, slots=True)
class ColdStartPhases:
    """Cold start, decomposed into the five phases RFC-0002 Method step 7
    requires reported separately (issue #56), plus the total — derived by
    :attr:`total`, never itself measured.

    ``session_creation`` and ``tokenizer_load`` are :class:`Metric` values
    that carry a real duration for the two embedding arms, which construct a
    session and load a tokenizer from the artefact respectively (issue #108)
    — and are ``None`` for ``overlap`` and ``fuzzy``, which construct
    neither at all: RFC-0002's Metrics table says these two phases are
    ``null`` for the classical arms, not zero, because the phase does not
    apply to them, rather than applying and costing nothing.
    """

    arm: str
    interpreter_start: Metric
    import_phase: Metric
    session_creation: Metric
    tokenizer_load: Metric
    first_inference: Metric
    not_attributable: frozenset[str]

    @property
    def total(self) -> Metric:
        return _sum_defined(
            [
                self.interpreter_start,
                self.import_phase,
                self.session_creation,
                self.tokenizer_load,
                self.first_inference,
            ]
        )


def measure_cold_start(
    arm: str,
    left_name: str,
    right_name: str,
    *,
    artifact: ArtifactRequirement | None = None,
) -> ColdStartPhases | Unavailable:
    """Cold start for ``arm``, decomposed into its five phases, in a fresh
    isolated worker (RFC-0002 Method step 7, issue #56).

    Interpreter start is timed across the process boundary: the parent
    records its own launch instant immediately before spawning, the worker
    records its as the first statement it executes, and the phase is their
    difference. Both readings use wall-clock time (``time.time()``), which is
    the one clock whose readings from two different processes on the same
    machine are comparable at all — ``time.perf_counter()``'s reference point
    is undefined across processes and is used for every other phase here,
    all of which are timed within the worker alone. Clamped at zero: process
    scheduling can occasionally make the difference read as a small negative
    number, and a negative duration is not a duration.
    """
    launch_epoch = time.time()
    params = {"arm": arm, "left_name": left_name, "right_name": right_name}
    result = _measure(arm, artifact, _COLD_START_SCRIPT, params)
    if isinstance(result, Unavailable):
        return result

    interpreter_start_seconds = max(
        0.0, cast(float, result["child_start_epoch"]) - launch_epoch
    )
    return ColdStartPhases(
        arm=arm,
        interpreter_start=Metric(
            value=interpreter_start_seconds, undefined_reason=None
        ),
        import_phase=Metric(
            value=cast(float, result["import_seconds"]), undefined_reason=None
        ),
        session_creation=_phase_or_reason(
            cast("float | None", result.get("session_creation_seconds")),
            _NO_SESSION_REASON,
        ),
        tokenizer_load=_phase_or_reason(
            cast("float | None", result.get("tokenizer_load_seconds")),
            _NO_TOKENIZER_REASON,
        ),
        first_inference=Metric(
            value=cast(float, result["first_inference_seconds"]), undefined_reason=None
        ),
        not_attributable=_NOT_ATTRIBUTABLE_PHASES,
    )


_NO_ARTIFACT_REASON = "classical arms carry no model artifact"


def measure_artifact_size(paths: Sequence[Path]) -> Metric:
    """Total bytes on disk across every artefact file an arm depends on
    (RFC-0002 Metrics, "Artefact size"; issue #63).

    Read directly off the filesystem via ``Path.stat().st_size`` at call
    time — never a constant compiled into this module, so the figure tracks
    whatever file is actually on disk rather than whatever a model card once
    said about it (issue #63's last bullet: "measured from the file, never
    from documentation about the model").

    ``paths`` empty means the arm carries no model artefact at all — true of
    both classical arms. That is recorded as an explicit undefined
    :class:`~joinless.evaluation.Metric`, the same "undefined is not zero"
    rule ADR-0013 states everywhere else in this module: ``0.0`` would claim
    a zero-byte artefact exists, when the classical arms have no artefact at
    all (issue #63: "the classical arms are not exempt... a zero or an empty
    cell for the classical arms is a fact worth stating explicitly"). This
    function does not spawn a worker — unlike every other metric in this
    module, a file's size on disk is not a runtime-resource figure a shared
    process could contaminate (module docstring), so no isolation is needed
    to make it mean what it says.
    """
    if not paths:
        return Metric(value=None, undefined_reason=_NO_ARTIFACT_REASON)
    total_bytes = sum(path.stat().st_size for path in paths)
    return Metric(value=float(total_bytes), undefined_reason=None)
