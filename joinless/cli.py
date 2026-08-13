# SPDX-License-Identifier: MIT
"""The ``joinless`` console entry point: ``resolve``, ``compare``, ``doctor``,
``benchmark``.

RFC-0003 names five commands; four exist at this point in the project (M2).
``report`` is later work and deliberately absent here and from ``--help`` — this
module's job is to name only the commands that actually exist, not to reserve a
place for ones that don't yet.

**The record-set schema for ``resolve``.** Neither the PRD nor RFC-0003 fixes a file
format for the two record sets FR-1 asks to be merged — that is a different object
from RFC-0005's labelled-pairs schema (``left_name``/``right_name``/``label``), which
describes evaluation pairs, not the entities ``resolve`` merges. This module reads
and writes **JSON Lines**: one JSON object per line, each carrying the fields
:class:`~joinless.records.Record` doesn't already know from context (``name``,
optionally ``latitude``/``longitude``/``fields``). A record's ``source`` is the
input file's stem and its ``ordinal`` is its position among that file's non-blank
lines, so a caller supplies nothing beyond the two files. JSON Lines needs no
dependency beyond the standard library, mirrors one row per entity the way the rest
of this project already reads corpora, and — being newline-delimited — never asks a
caller to hold an entire file in memory to parse it.

**The threshold flag lives in ``compare`` only** (RFC-0003 open question 2, and
issue #43's bullet). ``resolve`` therefore has no ``--threshold`` of its own: its
merge decision uses :data:`_DEFAULT_THRESHOLD`, a fixed value rather than a
calibrated one — ADR-0011's calibration procedure produces a threshold for a
*benchmark run*, evaluated against labelled pairs, and ``resolve`` runs over
un-labelled records with no ground truth to calibrate against. A conservative
default is used deliberately: a record that doesn't clear it stays unmatched
(auditable, recoverable) rather than folded into a merge nobody asked to verify.

**``doctor`` reports the installed profile without importing the runtime**
(ADR-0014, issue #44's bullet). :func:`importlib.util.find_spec` locates a module on
``sys.path`` without executing it, which is exactly the distinction ADR-0014 draws:
detecting *availability* must never cost the import boundary
``tests/test_import_boundary.py`` enforces.

**``benchmark`` runs RFC-0002's protocol over the built-in corpus and writes one
record** (issue #45). There is no ``--pairs`` flag and no labelled-pairs file loader
here — reading a supplied file is issue #76, milestone M7 — so the only input is
:mod:`joinless.corpus`'s synthetic corpus, pooled across every seed in
:data:`joinless.corpus.SEEDS` (:func:`_pool_corpora`) so that ADR-0011 rule 3's
"several deterministic seeds" is a fact about what threshold selection and the sealed
test actually drew from, not only about what
:func:`~joinless.runrecord.build_evaluation_set_identity` claims for the same run.
``overlap`` and ``fuzzy`` are always registered in :mod:`joinless.scoring`;
``embed-fp32`` is registered too (M3), but only *available* where
``JOINLESS_MODEL_CACHE_DIR`` names a directory holding its checksummed artefact
(:mod:`joinless.embedding`) — ``embed-int8`` remains unregistered, so ``get_scorer``
still raises ``ValueError`` for that one name, the same it raises for any other name
it does not recognise. All four names in :data:`_ARMS` are attempted regardless: an
unregistered name and a registered-but-unavailable one are both ADR-0013's "an arm
that cannot initialise is recorded... not omitted," satisfied by the real cases
rather than a stub.

Every command here is local computation over :mod:`joinless.records`,
:mod:`joinless.resolver`, :mod:`joinless.scoring`, :mod:`joinless.corpus`,
:mod:`joinless.evaluation`, :mod:`joinless.measurement` and :mod:`joinless.runrecord`,
plus stdlib file, platform and subprocess calls that never construct a socket —
``pmset`` (power mode) and the isolated workers :mod:`joinless.measurement` already
spawns are local process launches, not network I/O. Nothing in this module ever opens
a socket, so "the command completes with no network interface available" (all four
issues) holds structurally rather than by a check this module performs.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

from joinless import corpus
from joinless.corpus import Corpus, LabelledPair, Role
from joinless.evaluation import (
    Contradiction,
    EvaluationReport,
    ExpectedWinners,
    InvalidRun,
    SelectedThreshold,
    evaluate_sealed_test,
    find_contradictions,
    freeze_threshold,
    select_threshold,
)
from joinless.measurement import (
    Unavailable,
    measure_artifact_size,
    measure_cold_start,
    measure_peak_memory,
    measure_warm_latency,
)
from joinless.records import Record
from joinless.resolver import ResolutionResult
from joinless.resolver import resolve as resolve_records
from joinless.runrecord import (
    ArmResult,
    Environment,
    Hardware,
    Maybe,
    ModelIdentity,
    RunAssembly,
    RuntimeVersions,
    build_evaluation_set_identity,
    write_record,
)
from joinless.scoring import (
    ScorerUnavailable,
    ThresholdMatcher,
    get_artifact_paths,
    get_scorer,
)

# Not a calibrated value (ADR-0011's calibration procedure needs labelled pairs;
# neither `resolve` nor `compare`'s default has any). Deliberately conservative: for
# `resolve`, a merge nobody can audit is worse than a record left unmatched with a
# reason (issue #42); for `compare`, it gives the demonstration a defensible
# starting point that `--threshold` then moves (issue #43).
_DEFAULT_THRESHOLD = 0.8


def _read_records(path: Path, *, source: str) -> list[Record]:
    """Read one side of a ``resolve`` run: see the module docstring for the schema.

    Blank lines are skipped rather than counted, so a trailing newline at end of
    file — nearly universal in a hand- or editor-written JSON Lines file — cannot
    shift every later record's ``ordinal`` by one.
    """
    records: list[Record] = []
    ordinal = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row: dict[str, Any] = json.loads(line)
        records.append(
            Record(
                source=source,
                ordinal=ordinal,
                name=row["name"],
                latitude=row.get("latitude"),
                longitude=row.get("longitude"),
                fields=row.get("fields", {}),
            )
        )
        ordinal += 1
    return records


def _write_resolution(path: Path, result: ResolutionResult) -> None:
    """Write the merged set: every matched pair's merge, then every unmatched
    record with its reason (issue #42's bullet) — one JSON object per line.

    The two collections come straight from :class:`ResolutionResult`, which
    already separates them; this function only decides how each becomes one line.
    """
    lines: list[str] = []
    for pair in result.pairs:
        merged = pair.merged
        lines.append(
            json.dumps(
                {
                    "status": "matched",
                    "name": merged.name,
                    "latitude": merged.latitude,
                    "longitude": merged.longitude,
                    "fields": dict(merged.fields),
                    "sources": sorted(merged.sources),
                },
                sort_keys=True,
            )
        )
    for entry in result.unmatched:
        record = entry.record
        lines.append(
            json.dumps(
                {
                    "status": "unmatched",
                    "source": record.source,
                    "ordinal": record.ordinal,
                    "name": record.name,
                    "latitude": record.latitude,
                    "longitude": record.longitude,
                    "fields": dict(record.fields),
                    "reason": entry.reason,
                },
                sort_keys=True,
            )
        )
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def _cmd_resolve(args: argparse.Namespace) -> int:
    try:
        scorer = get_scorer(args.scorer)
    except (ValueError, ScorerUnavailable) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    left = _read_records(args.left, source=args.left.stem)
    right = _read_records(args.right, source=args.right.stem)
    matcher = ThresholdMatcher(scorer=scorer, threshold=_DEFAULT_THRESHOLD)
    result = resolve_records(left, right, matcher)
    _write_resolution(args.output, result)

    print(
        f"resolved {len(left)} left record(s) and {len(right)} right record(s) "
        f"under '{scorer.name}': {len(result.pairs)} matched pair(s), "
        f"{len(result.unmatched)} unmatched record(s) written to {args.output}"
    )
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    try:
        scorer = get_scorer(args.scorer)
    except (ValueError, ScorerUnavailable) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    matcher = ThresholdMatcher(scorer=scorer, threshold=args.threshold)

    # Timed as a single, isolated comparison — the illustrative figure RFC-0003
    # open question 3 draws a hard line around: it is printed once, never
    # repeated to make a difference perceptible, and never written anywhere.
    start = time.perf_counter()
    prepared_left = scorer.prepare(args.left_name)
    prepared_right = scorer.prepare(args.right_name)
    score = scorer.score(prepared_left, prepared_right)
    decision = matcher.matches(prepared_left, prepared_right)
    elapsed_ms = (time.perf_counter() - start) * 1000

    print(f"scorer: {scorer.name}")
    print(f"threshold: {args.threshold}")
    print(f"score: {score:.4f}")
    print(f"decision: {'match' if decision else 'no match'}")
    print(
        f"elapsed: {elapsed_ms:.4f} ms "
        "(illustrative timing for this single comparison only, "
        "never benchmark evidence — see `joinless benchmark`)"
    )
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    del args  # doctor takes none; every command in _COMMANDS shares this signature

    # find_spec locates a module on sys.path without executing it (ADR-0014):
    # the profile is checkable without paying, or even risking, the import cost
    # the boundary test polices.
    installed_profile = (
        "neural" if importlib.util.find_spec("onnxruntime") is not None else "base"
    )

    lines = [
        f"architecture: {platform.machine()}",
        f"operating system: {platform.system()} {platform.release()}",
        f"python version: {platform.python_version()}",
        f"joinless version: {importlib.metadata.version('joinless')}",
        # ADR-0006: this project never configures a GPU or NPU execution
        # provider, so the fact is stated rather than read off a live
        # inference session — asking a session would need the runtime import
        # this function exists to avoid.
        "execution provider: cpu (ADR-0006: no GPU or NPU provider is ever configured)",
        f"installed profile: {installed_profile}",
        "offline status: no command in this package opens a network connection",
        (
            "benchmark run record: none (doctor reports the environment, not a"
            " run; see benchmarks/ for run records)"
        ),
    ]
    print("\n".join(lines))
    return 0


# --- benchmark ---------------------------------------------------------------

# ADR-0008 names four arms. "overlap" and "fuzzy" are always registered;
# "embed-fp32" is registered too (M3) but only available where
# JOINLESS_MODEL_CACHE_DIR names a directory holding its checksummed artefact
# (joinless.embedding); "embed-int8" remains unregistered. All four are attempted
# regardless — see the module docstring for why an unregistered name and a
# registered-but-unavailable one are both the real ADR-0013 "unavailable" case
# rather than a stub.
_ARMS: tuple[str, ...] = ("overlap", "fuzzy", "embed-fp32", "embed-int8")

_WARMUP_COUNT = 5
_REPETITION_COUNT = 20

_SCHEMA = "benchmark-v2"

# ADR-0011 rule 4: "the expected winner per family is recorded before the run."
# Reasoning per family, grounded in joinless.corpus's own module docstring and
# joinless.scoring.OverlapScorer's documented weakness:
#   - word order: the overlap coefficient is a set intersection, invariant to token
#     order by construction.
#   - character noise / transliteration: OverlapScorer is documented as
#     character-blind — a single scrambled or transliterated character changes a
#     token entirely and removes any overlap. FuzzyScorer exists for exactly this
#     failure.
#   - semantic alias: built with zero shared tokens between the two names
#     (joinless.corpus docstring), so overlap's intersection is provably empty and it
#     always scores 0.0 — a classical arm is not the one "tempted" here (an
#     embedding arm is), and overlap is the classical arm guaranteed to reject it.
#   - near-miss negative: joinless.corpus's own docstring names this "the failure a
#     classical, string-based arm is expected to be tempted by" — both overlap and
#     fuzzy are expected to over-match here, so the arm expected to get it right is
#     embed-fp32. Naming it anyway keeps the pre-registration honest even where
#     embed-fp32 has no artefact to load (JOINLESS_MODEL_CACHE_DIR unset) and
#     contributes no row: find_contradictions then compares overlap against fuzzy
#     alone, and whichever of the two "wins" is correctly a contradiction — the arm
#     expected to win this family did not run, not "the wrong classical arm won."
#   - everything else (exact, formatting, abbreviation): RFC-0002's evaluation-set
#     table marks these as not expected to separate the arms at all; overlap is
#     named as the representative baseline.
_EXPECTED_WINNERS = ExpectedWinners(
    winners={
        "exact": "overlap",
        "formatting": "overlap",
        "word order": "overlap",
        "abbreviation": "overlap",
        "character noise": "fuzzy",
        "semantic alias": "overlap",
        "transliteration": "fuzzy",
        "near-miss negative": "embed-fp32",
    }
)


def _parse_pmset_output(text: str) -> str:
    """Normalise macOS ``pmset -g batt`` output to ``"ac"``, ``"battery"`` or
    ``"unknown"``."""
    if "AC Power" in text:
        return "ac"
    if "Battery Power" in text:
        return "battery"
    return "unknown"


def _parse_linux_power_supply_status(text: str) -> str:
    """Normalise the contents of a ``/sys/class/power_supply/*/status`` file the
    same way."""
    normalised = text.strip().lower()
    if normalised in {"charging", "full"}:
        return "ac"
    if normalised == "discharging":
        return "battery"
    return "unknown"


# A module-level name, rather than a literal inlined into _detect_power_mode, so a
# test can point it at a real temporary directory instead of a path that exists on
# Linux alone (ADR-0016 rule 3: use the real thing rather than mocking the
# filesystem).
_LINUX_POWER_SUPPLY_DIR = Path("/sys/class/power_supply")


def _detect_linux_power_mode(supply_dir: Path) -> str:
    """The first power-supply status file under ``supply_dir``, normalised — or
    ``"unknown"`` when none exists."""
    status_files = sorted(supply_dir.glob("*/status"))
    if not status_files:
        return "unknown"
    return _parse_linux_power_supply_status(status_files[0].read_text(encoding="utf-8"))


def _detect_power_mode() -> str:
    """``"ac"``, ``"battery"`` or ``"unknown"`` (RFC-0002 Method step 5), read via
    ``pmset`` on Darwin and ``/sys/class/power_supply`` on Linux (ADR-0002's
    macOS-and-Linux scope) — neither of which opens a socket."""
    system = platform.system()
    if system == "Darwin":
        result = subprocess.run(
            ["pmset", "-g", "batt"], capture_output=True, text=True, check=False
        )
        return _parse_pmset_output(result.stdout)
    if system == "Linux":
        return _detect_linux_power_mode(_LINUX_POWER_SUPPLY_DIR)
    return "unknown"


def _hardware() -> Hardware:
    return Hardware(
        cpu_count=os.cpu_count() or 1,
        machine=platform.machine(),
        python_version=platform.python_version(),
        release=platform.release(),
        system=platform.system(),
        total_memory_bytes=int(os.sysconf("SC_PAGE_SIZE"))
        * int(os.sysconf("SC_PHYS_PAGES")),
    )


def _model_identity(arm: str) -> ModelIdentity | None:
    """Which model ``arm`` loaded this run, or ``None`` for an arm that
    carries no model at all — ``overlap`` and ``fuzzy`` (ADR-0003), and any
    arm whose own ``get_scorer`` call has not already succeeded, which the
    caller (:func:`_measure_arm`, the only one) is responsible for having
    established before reaching here.

    Imported lazily, at call time — mirroring :mod:`joinless.scoring`'s own
    lazy reach into :mod:`joinless.embedding` (ADR-0014, ADR-0017) — so a
    classical-only run, or a run in which ``embed-fp32`` could not
    initialise, never imports :mod:`joinless.embedding` from here either.
    """
    if arm == "embed-fp32":
        from joinless import embedding

        return embedding.model_identity_fp32()
    return None


def _runtime_versions(*, model_identity: ModelIdentity | None) -> RuntimeVersions:
    """``rapidfuzz`` is a base dependency (ADR-0014) so its version is always
    read from installed-package metadata. ``onnxruntime`` is read a different
    way, and only when ``model_identity`` says a neural arm actually loaded
    one: :data:`sys.modules` already holds the exact module that arm's own
    ``get_scorer`` call imported (:mod:`joinless.embedding`'s
    ``load_fp32_scorer``), so ``onnxruntime.__version__`` names the runtime
    this run actually used rather than whatever a separate metadata lookup
    happens to find on the path. A run with no successful neural arm keeps
    the version inapplicable (ADR-0013) — never absent-and-uninteresting,
    and never guessed at from a package that was never imported.
    """
    if model_identity is not None:
        import onnxruntime

        onnxruntime_version = Maybe(value=onnxruntime.__version__, reason=None)
    else:
        onnxruntime_version = Maybe(value=None, reason="no neural arm in this run")
    return RuntimeVersions(
        onnxruntime=onnxruntime_version,
        rapidfuzz=importlib.metadata.version("rapidfuzz"),
    )


def _environment(
    power_mode: str, *, model_identity: ModelIdentity | None
) -> Environment:
    if model_identity is not None:
        model = Maybe(value=model_identity, reason=None)
    else:
        model = Maybe(value=None, reason="no neural arm in this run")
    return Environment(
        hardware=_hardware(),
        runtime_versions=_runtime_versions(model_identity=model_identity),
        power_mode=power_mode,
        # Single-threaded: no arm this module can run configures a thread pool or a
        # GPU/NPU execution provider (ADR-0006), and neither classical scorer
        # spawns a worker thread of its own.
        thread_count=1,
        warmup_count=_WARMUP_COUNT,
        repetition_count=_REPETITION_COUNT,
        model=model,
        # embed-int8 remains unregistered (module docstring), so no run can yet
        # produce a quantized-operator list.
        quantized_operators=Maybe(value=None, reason="no int8 arm in this run"),
    )


def _pool_corpora(corpora: Sequence[Corpus]) -> Corpus:
    """Combine every seed's pairs and roles into one corpus (ADR-0011 rule 3: "the
    corpus is generated under several deterministic seeds").
    :func:`~joinless.evaluation.select_threshold` and
    :func:`~joinless.evaluation.evaluate_sealed_test` each take one
    :class:`~joinless.corpus.Corpus`; pooling here is what lets a single
    threshold-selection and sealed-test pass draw from every seed rather than only
    the first, matching the seeds
    :func:`~joinless.runrecord.build_evaluation_set_identity` records for the same
    run. Every seed's pair ids already carry that seed as a prefix
    (:mod:`joinless.corpus`), so no id collides across corpora and ``Corpus``'s own
    duplicate check (its ``__post_init__``) passes.

    ``seed`` on the returned corpus is the first corpus's seed — a field this
    function has to fill in to construct a valid ``Corpus``, but nothing downstream
    reads it: the run record's evaluation-set identity is built from ``corpora``
    directly (:func:`~joinless.runrecord.build_evaluation_set_identity`), not from
    this pooled value.
    """
    pairs: list[LabelledPair] = []
    roles: dict[str, Role] = {}
    for one_corpus in corpora:
        pairs.extend(one_corpus.pairs)
        roles.update(one_corpus.roles)
    return Corpus(
        seed=corpora[0].seed, pairs=tuple(pairs), roles=MappingProxyType(roles)
    )


def _measure_arm(
    arm: str, pooled: Corpus, left_name: str, right_name: str, power_mode: str
) -> tuple[ArmResult, SelectedThreshold | None, ModelIdentity | None]:
    """Run the full protocol for ``arm``, or mark it unavailable everywhere at
    once.

    Every field this returns is decided by exactly one ``get_scorer`` call:
    succeeding runs threshold selection, sealed-test evaluation and all four
    resource measurements; failing marks accuracy, warm latency, peak memory,
    cold start and artifact size unavailable with that one call's reason,
    without spawning a worker to rediscover the same failure a second way —
    ADR-0013's rule is that the arm is recorded, not how many times its
    unavailability gets independently confirmed. No threshold is selected for
    an arm that never got this far, so it contributes nothing to the run's
    ``selected_thresholds`` list. The third element is what ``arm`` loaded as
    a model, if anything (:func:`_model_identity`) — ``None`` for the same
    "never got this far" arms, and for every classical arm regardless.
    """
    try:
        scorer = get_scorer(arm)
    except (ValueError, ScorerUnavailable) as exc:
        reason = str(exc)
        return (
            ArmResult(
                accuracy=InvalidRun(reason=reason),
                warm_latency=Unavailable(arm=arm, reason=reason),
                peak_memory=Unavailable(arm=arm, reason=reason),
                cold_start=Unavailable(arm=arm, reason=reason),
                artifact_size=Unavailable(arm=arm, reason=reason),
            ),
            None,
            None,
        )

    selected = select_threshold(pooled, scorer)
    frozen = freeze_threshold(selected)
    accuracy = evaluate_sealed_test(pooled, scorer, frozen)
    warm_latency = measure_warm_latency(
        arm,
        left_name,
        right_name,
        warmup_count=_WARMUP_COUNT,
        repetition_count=_REPETITION_COUNT,
    )
    peak_memory = measure_peak_memory(arm, left_name, right_name, power_mode=power_mode)
    cold_start = measure_cold_start(arm, left_name, right_name)
    artifact_size = measure_artifact_size(get_artifact_paths(arm))
    return (
        ArmResult(
            accuracy=accuracy,
            warm_latency=warm_latency,
            peak_memory=peak_memory,
            cold_start=cold_start,
            artifact_size=artifact_size,
        ),
        selected,
        _model_identity(arm),
    )


def _find_contradictions(
    expected: ExpectedWinners, arm_results: Mapping[str, ArmResult]
) -> tuple[Contradiction, ...]:
    """Compare ``expected`` against every arm's actual per-family accuracy
    (ADR-0011 rule 4, issue #50), delegating the comparison itself to
    :func:`joinless.evaluation.find_contradictions` — this function's own job
    is only to pick out which arms have a real per-family table to compare.
    An arm whose ``accuracy`` is :class:`~joinless.evaluation.InvalidRun`
    (an unregistered scorer, or one whose dependency or artefact is
    unavailable) contributes no row, which is what leaves
    ``find_contradictions``'s own "fewer than two comparable arms" rule to
    skip a family such as "near-miss negative" wherever its expected winner
    is an arm that did not run in this environment (module docstring) —
    a fact this function does not re-decide, only feeds correctly.
    """
    reports: dict[str, EvaluationReport] = {
        arm: result.accuracy
        for arm, result in arm_results.items()
        if isinstance(result.accuracy, EvaluationReport)
    }
    return find_contradictions(expected, reports)


def _format_contradictions(contradictions: Sequence[Contradiction]) -> list[str]:
    """The lines ``benchmark`` prints for its contradictions finding (issue
    #50: "findings... not a footnote") — a pure function of the comparison
    :func:`_find_contradictions` already computed, so what the command prints
    and what it persists in the run record's ``contradictions`` field can
    never say two different things about the same run.
    """
    if not contradictions:
        return ["contradictions: none — every pre-registered expectation held"]
    lines = [
        (
            f"contradictions: {len(contradictions)} pre-registered expectation(s) "
            "did not hold"
        )
    ]
    for contradiction in contradictions:
        actual = ", ".join(repr(arm) for arm in contradiction.actual_winners)
        lines.append(
            f"  {contradiction.family}: expected {contradiction.expected_winner!r} "
            f"to win, actual winner(s): {actual}"
        )
    return lines


def _cmd_benchmark(args: argparse.Namespace) -> int:
    del args  # benchmark takes no flags: the built-in corpus, every arm, one record.

    started_at = datetime.now(UTC)
    power_mode = _detect_power_mode()
    corpora = corpus.generate_corpora(corpus.SEEDS)
    pooled = _pool_corpora(corpora)
    evaluation_set = build_evaluation_set_identity(corpora)
    left_name = pooled.pairs[0].left_name
    right_name = pooled.pairs[0].right_name

    assembly = RunAssembly(expected_winners=_EXPECTED_WINNERS)
    arm_results: dict[str, ArmResult] = {}
    selected_thresholds: list[SelectedThreshold] = []
    # At most one arm on this branch ever loads a model (embed-fp32; embed-int8
    # remains unregistered), so "the last one that reported an identity" and
    # "the one that did" are the same arm — this will need revisiting once a
    # second neural arm is registered and the two could disagree.
    model_identity: ModelIdentity | None = None
    for arm in _ARMS:
        result, selected, identity = _measure_arm(
            arm, pooled, left_name, right_name, power_mode
        )
        assembly.add_arm(arm, result)
        arm_results[arm] = result
        if selected is not None:
            selected_thresholds.append(selected)
        if identity is not None:
            model_identity = identity

    # ADR-0011 rule 4 / issue #50: the pre-registered expectations are compared
    # against the actual outcome here, once, and the same value is both printed
    # below and persisted on the record - never recomputed for either.
    contradictions = _find_contradictions(_EXPECTED_WINNERS, arm_results)

    record = assembly.build(
        schema=_SCHEMA,
        started_at=started_at,
        command=("joinless", "benchmark"),
        environment=_environment(power_mode, model_identity=model_identity),
        evaluation_set=evaluation_set,
        selected_thresholds=tuple(selected_thresholds),
        contradictions=contradictions,
    )
    try:
        path = write_record(record, Path("benchmarks"))
    except FileExistsError as exc:
        print(
            f"a run record already exists at {exc.filename}: refusing to overwrite it",
            file=sys.stderr,
        )
        return 1
    print(f"wrote {path}")
    for line in _format_contradictions(contradictions):
        print(line)
    return 0


_COMMANDS: Mapping[str, Callable[[argparse.Namespace], int]] = {
    "resolve": _cmd_resolve,
    "compare": _cmd_compare,
    "doctor": _cmd_doctor,
    "benchmark": _cmd_benchmark,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="joinless",
        description="Keyless entity resolution: measurement and reproduction surface.",
    )
    subparsers = parser.add_subparsers(dest="command")

    resolve_parser = subparsers.add_parser(
        "resolve",
        help="Merge two record sets into one, and write it out.",
        description=(
            "Read two JSON Lines record sets, resolve them under the chosen "
            "scoring arm, and write one merged JSON Lines set: matched pairs "
            "merged under the FR-5 policy, and every unmatched record kept "
            "alongside the reason it did not match."
        ),
    )
    resolve_parser.add_argument(
        "--left", required=True, type=Path, help="JSON Lines file: the left record set."
    )
    resolve_parser.add_argument(
        "--right",
        required=True,
        type=Path,
        help="JSON Lines file: the right record set.",
    )
    resolve_parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Where to write the merged JSON Lines set.",
    )
    resolve_parser.add_argument(
        "--scorer",
        default="overlap",
        # No `choices=`: the set of known arms lives in exactly one place,
        # joinless.scoring's own registry, and get_scorer already reports an
        # unknown name with the full available list. Duplicating that list here
        # would be a second copy this module would have to remember to update.
        help="Named scoring arm (default: %(default)s).",
    )

    compare_parser = subparsers.add_parser(
        "compare",
        help="Score one name pair under a chosen arm and print the decision.",
        description=(
            "Score exactly one name pair and print its score, its decision "
            "under the given threshold, and an illustrative timing. Never "
            "writes a run record — see `joinless benchmark` for measurement."
        ),
    )
    compare_parser.add_argument("left_name", help="The first name.")
    compare_parser.add_argument("right_name", help="The second name.")
    compare_parser.add_argument(
        "--scorer", default="overlap", help="Named scoring arm (default: %(default)s)."
    )
    compare_parser.add_argument(
        "--threshold",
        type=float,
        default=_DEFAULT_THRESHOLD,
        help="Decision threshold (default: %(default)s). The only command with this flag.",
    )

    subparsers.add_parser(
        "doctor",
        help="Report the execution environment.",
        description=(
            "Report architecture, execution provider, installed profile and "
            "offline status, in a form that can be pasted directly into a "
            "bug report."
        ),
    )

    subparsers.add_parser(
        "benchmark",
        help="Run RFC-0002's protocol over the built-in corpus and write a record.",
        description=(
            "Run every configured arm over the built-in synthetic corpus under "
            "RFC-0002's protocol, and write one record to benchmarks/. An arm "
            "that cannot initialise is recorded as unavailable with a reason, "
            "never omitted. No flags: the corpus and the arms are fixed."
        ),
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command is None:
        return 0
    return _COMMANDS[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
