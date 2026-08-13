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

from joinless.evaluation import Metric


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

    def _percentile(data, fraction):
        ordered = sorted(data)
        return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]

    print(json.dumps({
        "status": "ok",
        "p50_seconds": _percentile(durations, 0.50),
        "p99_seconds": _percentile(durations, 0.99),
    }))
"""


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


# The one prepare call before either timed section runs past runtime start-up
# cost (import, lazy graph load) — measured separately by measure_cold_start,
# so folding it into hoisted or naive here would double-count it (RFC-0002
# Method step 2). Both timed sections share this single warm-up rather than
# each getting their own, so neither path is warmed up more than the other.
_PREPARATION_COST_SCRIPT = """
import json
import os
import time

import joinless.scoring as scoring

params = json.loads(os.environ["JOINLESS_MEASURE_PARAMS"])
try:
    scorer = scoring.get_scorer(params["arm"])
except (ValueError, scoring.ScorerUnavailable) as exc:
    print(json.dumps({"status": "unavailable", "reason": str(exc)}))
else:
    left_names = params["left_names"]
    right_names = params["right_names"]
    left_indices = params["left_indices"]
    right_indices = params["right_indices"]

    scorer.prepare(left_names[0])

    _start = time.perf_counter()
    scorer.prepare_all(left_names)
    scorer.prepare_all(right_names)
    hoisted_seconds = time.perf_counter() - _start

    _start = time.perf_counter()
    for _left_index, _right_index in zip(left_indices, right_indices):
        scorer.prepare(left_names[_left_index])
        scorer.prepare(right_names[_right_index])
    naive_seconds = time.perf_counter() - _start

    print(json.dumps({
        "status": "ok",
        "hoisted_seconds": hoisted_seconds,
        "naive_seconds": naive_seconds,
    }))
"""


@dataclass(frozen=True, slots=True)
class PreparationCost:
    """Hoisted and naive preparation cost for one arm, over the same
    candidate set whose bucket occupancy the run record carries alongside it
    (issue #66) — never quoted without the occupancy that produced it
    (ADR-0009: "a hoist speed-up quoted without the occupancy that produced
    it cannot be transferred to any other dataset").

    ``hoisted_seconds`` times ``prepare_all`` called once per side, batched
    — the production pattern ADR-0009 requires. ``naive_seconds`` times
    ``prepare`` called fresh for both records of every comparison in
    ``comparison_count``, reproducing the redundant recomputation the hoist
    removes. Both are timed in the same isolated worker (RFC-0002 Method
    step 7 names "preparation cost" alongside cold start, warm scoring and
    peak RSS as a metric requiring isolation), sharing one warm-up so
    neither path is measured any warmer than the other.

    ``record_count`` and ``comparison_count`` travel with the figures for
    the same reason ``WarmLatency`` carries its own ``warmup_count`` and
    ``repetition_count``: a reader holding one ``PreparationCost`` alone can
    see what dataset produced it without cross-referencing another part of
    the run record.
    """

    arm: str
    hoisted_seconds: float
    naive_seconds: float
    record_count: int
    comparison_count: int


def measure_preparation_cost(
    arm: str,
    left_names: Sequence[str],
    right_names: Sequence[str],
    comparison_pairs: Sequence[tuple[int, int]],
    *,
    artifact: ArtifactRequirement | None = None,
) -> PreparationCost | Unavailable:
    """Hoisted vs naive preparation cost for ``arm`` over
    ``comparison_pairs`` — each pair a ``(left_index, right_index)`` into
    ``left_names``/``right_names`` — in a fresh isolated worker (RFC-0002
    Method step 7, issue #66).

    ``left_names``, ``right_names`` and ``comparison_pairs`` must each be
    non-empty — validated before anything is spawned, so a caller error is a
    ``ValueError`` here rather than a confusing shape from the worker.
    """
    if not left_names:
        raise ValueError("left_names must not be empty")
    if not right_names:
        raise ValueError("right_names must not be empty")
    if not comparison_pairs:
        raise ValueError("comparison_pairs must not be empty")

    params = {
        "arm": arm,
        "left_names": list(left_names),
        "right_names": list(right_names),
        "left_indices": [pair[0] for pair in comparison_pairs],
        "right_indices": [pair[1] for pair in comparison_pairs],
    }
    result = _measure(arm, artifact, _PREPARATION_COST_SCRIPT, params)
    if isinstance(result, Unavailable):
        return result
    return PreparationCost(
        arm=arm,
        hoisted_seconds=cast(float, result["hoisted_seconds"]),
        naive_seconds=cast(float, result["naive_seconds"]),
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
    everything the worker did, by construction, for whichever arm runs.
    Today that is verified only for the classical arms, and neither one
    loads a model (ADR-0014; the embedding arms that would are milestone
    M3), so "including model load" is not yet exercised by any test — it
    follows from the sampling point coming after everything the arm's
    worker does, not from having measured a model-loading arm.

    ``thread_count`` and ``power_mode`` travel with the figure because both
    move it (issue #55's third bullet): ``power_mode`` is supplied by the
    caller rather than detected here — reading a platform's power state is a
    platform-specific concern no issue in this batch asks this module to
    build, so the field exists to carry a value the caller already has,
    not to derive one.
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

try:
    scorer = scoring.get_scorer(params["arm"])
except (ValueError, scoring.ScorerUnavailable) as exc:
    print(json.dumps({"status": "unavailable", "reason": str(exc)}))
    sys.exit(0)

_before_first = time.perf_counter()
left = scorer.prepare(params["left_name"])
right = scorer.prepare(params["right_name"])
scorer.score(left, right)
_after_first = time.perf_counter()

print(json.dumps({
    "status": "ok",
    "child_start_epoch": _child_start,
    "import_seconds": _after_import - _before_import,
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
    that are ``None`` for every arm this module can measure today: only the
    classical arms initialise (ADR-0014), and RFC-0002's Metrics table says
    those two phases are ``null`` for them, not zero, because a classical
    arm constructs no session and loads no tokenizer at all — the phase does
    not apply, rather than applying and costing nothing.
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
        session_creation=Metric(value=None, undefined_reason=_NO_SESSION_REASON),
        tokenizer_load=Metric(value=None, undefined_reason=_NO_TOKENIZER_REASON),
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
