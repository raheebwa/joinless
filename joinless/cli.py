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
``embed-fp32`` (M3) and ``embed-int8`` (issue #67, on RFC-0004's spike record
recording a "go") are registered too, each *available* only where
``JOINLESS_MODEL_CACHE_DIR`` names a directory holding its own checksummed
artefact (:mod:`joinless.embedding`) — the int8 arm's own model file, plus the
fp32 arm's tokenizer, which the two arms share. All four names in :data:`_ARMS`
are attempted regardless of availability: a registered-but-unavailable arm is
ADR-0013's "an arm that cannot initialise is recorded... not omitted," and every
arm here is checked the same way, through :func:`joinless.scoring.get_scorer`'s
probe-then-factory order — there is no unregistered name left among them for a
caller to hit ``ValueError`` on.

**The int8 arm's matmul-conversion census is read from its graph, once, at the end
of the arm loop** (:func:`_quantized_operators`, issue #68) — never copied from
``benchmarks/20260812T181752Z-quantization-spike.json``, and checked against what
that record established for this exact artefact, both which replacement operator
types are present
(:data:`joinless.embedding.INT8_QUANTIZED_OPERATORS`) and how many of each
candidate operator type converted
(:data:`joinless.embedding.INT8_MATMUL_CONVERSION`) — the counts a reader needs to
tell a partly-quantized encoder's expected, unchanged latency apart from a defect
(issue #68 finding 1). A graph whose live-read census does not equal either pinned
expectation raises :class:`joinless.embedding.QuantizedOperatorMismatchError`,
which :func:`_cmd_benchmark` lets abort the whole run rather than folding into one
arm's ``ArmResult`` the way an ordinary checksum mismatch does (ADR-0013 rule 3):
no record is produced for a run whose understanding of what it quantized cannot be
trusted, not a record with one row quietly marked unavailable.

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
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

from joinless import corpus
from joinless.corpus import Corpus, LabelledPair, Role
from joinless.evaluation import (
    AccuracyDivergence,
    Contradiction,
    EvaluationReport,
    ExpectedWinners,
    InvalidRun,
    Metric,
    SelectedThreshold,
    compute_accuracy_divergence,
    evaluate_sealed_test,
    find_contradictions,
    freeze_threshold,
    select_threshold,
)
from joinless.measurement import (
    PreparationCost,
    Unavailable,
    measure_artifact_size,
    measure_cold_start,
    measure_peak_memory,
    measure_preparation_cost,
    measure_warm_latency,
)
from joinless.records import Record
from joinless.resolver import (
    DEFAULT_CELL_SIZE_DEGREES,
    PreparationPath,
    ResolutionResult,
    bucket_occupancy,
    candidate_pairs,
    score_candidates,
)
from joinless.resolver import resolve as resolve_records
from joinless.runrecord import (
    ArmResult,
    BucketOccupancy,
    Environment,
    Hardware,
    MatmulConversion,
    Maybe,
    ModelIdentity,
    PreparationAsymmetry,
    PreparationComparison,
    RunAssembly,
    RuntimeVersions,
    build_evaluation_set_identity,
    write_record,
)
from joinless.scoring import (
    Scorer,
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

# ADR-0008 names four arms, all four registered in joinless.scoring: "overlap" and
# "fuzzy" unconditionally, "embed-fp32" (M3) and "embed-int8" (issue #67) each
# available only where JOINLESS_MODEL_CACHE_DIR names a directory holding its own
# checksummed artefact (joinless.embedding). All four are attempted regardless —
# see the module docstring for why a registered-but-unavailable arm is the real
# ADR-0013 "unavailable" case rather than a stub.
_ARMS: tuple[str, ...] = ("overlap", "fuzzy", "embed-fp32", "embed-int8")

_WARMUP_COUNT = 5
_REPETITION_COUNT = 20

# v2 -> v3: three breaking shape changes to a record other tooling may parse,
# mirroring the precedent that moved v1 -> v2 when `contradictions[].actual_winner`
# was renamed and retyped (a breaking shape change to a persisted field moves the
# schema tag with it). Here: `environment.model` (a single nullable model identity)
# became `environment.models` (a mapping keyed by arm name); a new top-level
# `int8_accuracy_divergence` field was added; and `environment.quantized_operators`
# was retyped from a flat operator-type list to a matmul-conversion census keyed by
# candidate operator type (issue #68 finding 1).
#
# v3 -> v4: `results.<arm>.preparation` was added (issue #65's third bullet) - each
# arm's naive and hoisted preparation paths, each self-tagged with the path that
# produced it. A new required field on every arm's result is exactly the kind of
# breaking shape change the v2 -> v3 comment above already names the policy for.
#
# v4 -> v5: three more additions, all issue #66 - `results.<arm>.preparation_cost`
# (hoisted/naive preparation *time*, as distinct from v4's score-equality-only
# `preparation`), and two new top-level fields, `bucket_occupancy` (the candidate-
# bucket occupancy distribution the run's preparation-cost sample was drawn from)
# and `preparation_asymmetry` (the classical/neural hoist speed-up comparison,
# reported as a result rather than left for a reader to derive). A new required
# field anywhere in the record is the same breaking shape change the v2 -> v3
# comment above already names the policy for.
_SCHEMA = "benchmark-v4"

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
    classical-only run, or a run in which neither neural arm could
    initialise, never imports :mod:`joinless.embedding` from here either.
    """
    if arm == "embed-fp32":
        from joinless import embedding

        return embedding.model_identity_fp32()
    if arm == "embed-int8":
        from joinless import embedding

        return embedding.model_identity_int8()
    return None


def _runtime_versions(*, models: Mapping[str, ModelIdentity]) -> RuntimeVersions:
    """``rapidfuzz`` is a base dependency (ADR-0014) so its version is always
    read from installed-package metadata. ``onnxruntime`` is read a different
    way, and only when ``models`` is non-empty — at least one neural arm
    actually loaded one: :data:`sys.modules` already holds the exact module
    whichever arm's own ``get_scorer`` call imported
    (:mod:`joinless.embedding`'s ``load_fp32_scorer``/``load_int8_scorer``,
    both of which import the same top-level ``onnxruntime`` package), so
    ``onnxruntime.__version__`` names the runtime this run actually used
    rather than whatever a separate metadata lookup happens to find on the
    path. A run with no successful neural arm keeps the version inapplicable
    (ADR-0013) — never absent-and-uninteresting, and never guessed at from a
    package that was never imported.
    """
    if models:
        import onnxruntime

        onnxruntime_version = Maybe(value=onnxruntime.__version__, reason=None)
    else:
        onnxruntime_version = Maybe(value=None, reason="no neural arm in this run")
    return RuntimeVersions(
        onnxruntime=onnxruntime_version,
        rapidfuzz=importlib.metadata.version("rapidfuzz"),
    )


def _quantized_operators(
    models: Mapping[str, ModelIdentity],
) -> Maybe[Mapping[str, MatmulConversion]]:
    """The int8 arm's matmul-conversion census for this run, read from its own
    graph — or an explicit reason it does not apply (issue #68's first and second
    bullets, finding 1: "how many of the graph's matmuls were converted and how
    many remain in fp32", not a bare list of the operator types present).

    RFC-0002 Method step 5 lists "quantized operator list" in the same clause as
    "model identity, revision and checksum" — the fact this function reads
    ``models`` for, keyed the same way :class:`~joinless.runrecord.Environment`
    already keys model identity by arm. It stays a single field rather than a
    second arm-keyed mapping: ADR-0007 fixes v1 at exactly one quantization pass,
    so ``embed-int8`` is the only arm this can ever apply to, and a mapping with
    one possible key would be structure this project does not need yet (YAGNI).

    ``None`` with a reason when no int8 arm loaded a model in this run (ADR-0013)
    — there is no graph to read anything from. Otherwise reads the graph fresh, at
    call time, via :func:`joinless.embedding.verify_int8_operators` — never the
    spike record, and never asserted from ``models``'s own checksum, which
    :func:`joinless.embedding.probe_int8` already verified but says nothing about
    which operators the checksummed bytes actually contain.

    Raises :class:`joinless.embedding.QuantizedOperatorMismatchError` uncaught —
    :func:`_cmd_benchmark` is the one that decides what "the graph does not match
    its recorded operator list" means for the run as a whole (issue #68's third
    bullet): refusing to write any record, not marking one arm unavailable while
    the rest of the record is written as if nothing were wrong. That guard now
    covers the matmul-conversion census too, not only which operator types are
    present (:func:`joinless.embedding.verify_int8_operators`'s own docstring).

    Imported lazily, at call time, mirroring :func:`_model_identity` — so a run
    with no int8 arm never imports :mod:`joinless.embedding` from here either.
    """
    if "embed-int8" not in models:
        return Maybe(value=None, reason="no int8 arm in this run")

    from joinless import embedding

    model_path, _ = embedding.resolve_int8_model_paths(os.environ)
    _, census = embedding.verify_int8_operators(model_path)
    return Maybe(value=census, reason=None)


# issue #65's third bullet, Finding 2: `evaluate_sealed_test` (via `_predict`)
# and this module's own warm-latency worker both prepare once, ahead of scoring,
# and neither has a naive counterpart a run could select instead - a structural
# fact about those two call sites, not a per-run choice, so it is a constant
# here rather than a value threaded through _environment's own parameters. See
# Environment.measurement_preparation_path's docstring for which figures this
# does, and does not, cover.
_MEASUREMENT_PREPARATION_PATH: PreparationPath = "hoisted"


def _environment(
    power_mode: str,
    *,
    models: Mapping[str, ModelIdentity],
    quantized_operators: Maybe[Mapping[str, MatmulConversion]],
) -> Environment:
    return Environment(
        hardware=_hardware(),
        runtime_versions=_runtime_versions(models=models),
        power_mode=power_mode,
        # Single-threaded: no arm this module can run configures a thread pool or a
        # GPU/NPU execution provider (ADR-0006), and neither classical scorer
        # spawns a worker thread of its own.
        thread_count=1,
        warmup_count=_WARMUP_COUNT,
        repetition_count=_REPETITION_COUNT,
        models=models,
        quantized_operators=quantized_operators,
        measurement_preparation_path=_MEASUREMENT_PREPARATION_PATH,
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


# The candidate-bucket occupancy this run's shared preparation-cost sample
# produces (issue #66): one group of records per grid cell, deliberately
# uneven sizes so bucket_occupancy() reports a real distribution rather than
# one repeated value - a cell with 1 record, one with 2, one with 3, one with
# 4. Every left/right record in a group shares one coordinate, so candidate
# generation pairs a group's records with every other record in that same
# group and none from any other group (groups sit whole integer degrees
# apart, far outside DEFAULT_CELL_SIZE_DEGREES's neighbourhood).
_PREPARATION_OCCUPANCY_GROUPS: tuple[int, ...] = (1, 2, 3, 4)
_PREPARATION_SAMPLE_SIZE = sum(_PREPARATION_OCCUPANCY_GROUPS)


@dataclass(frozen=True, slots=True)
class _PreparationSample:
    """The one candidate set every registered arm's preparation cost is
    measured over in a run (issue #66) — the same set
    :attr:`occupancy` describes, so a cost figure is never quoted without
    the occupancy that produced it (ADR-0009).

    ``comparison_pairs`` is ``(left_index, right_index)`` into
    ``left_names``/``right_names`` — indices rather than the
    :class:`~joinless.records.Record` objects themselves, because this is
    exactly the shape :func:`~joinless.measurement.measure_preparation_cost`
    passes to its isolated worker over an environment variable, which can
    carry a list of plain values but not a ``Record``.
    """

    left_names: tuple[str, ...]
    right_names: tuple[str, ...]
    comparison_pairs: tuple[tuple[int, int], ...]
    occupancy: BucketOccupancy


def _build_preparation_sample(pooled: Corpus) -> _PreparationSample:
    """Build this run's shared preparation-cost sample from the pooled
    corpus's own real pair names (issue #66) — grouped by
    :data:`_PREPARATION_OCCUPANCY_GROUPS` into ``DEFAULT_CELL_SIZE_DEGREES``
    cells of deliberately uneven size, then run through the resolver's own
    :func:`~joinless.resolver.candidate_pairs` and
    :func:`~joinless.resolver.bucket_occupancy` — the same grid blocking a
    real :func:`~joinless.resolver.resolve` run would use, so this sample's
    occupancy is a fact about that blocking, not a number this function
    invents and calls occupancy.
    """
    if len(pooled.pairs) < _PREPARATION_SAMPLE_SIZE:
        raise ValueError(
            "pooled corpus has fewer pairs than the preparation sample needs "
            f"({len(pooled.pairs)} < {_PREPARATION_SAMPLE_SIZE})"
        )

    left_records: list[Record] = []
    right_records: list[Record] = []
    index = 0
    for group, size in enumerate(_PREPARATION_OCCUPANCY_GROUPS):
        coordinate = float(group)
        for _ in range(size):
            pair = pooled.pairs[index]
            left_records.append(
                Record(
                    source="preparation-sample",
                    ordinal=index,
                    name=pair.left_name,
                    latitude=coordinate,
                    longitude=coordinate,
                )
            )
            right_records.append(
                Record(
                    source="preparation-sample",
                    ordinal=index,
                    name=pair.right_name,
                    latitude=coordinate,
                    longitude=coordinate,
                )
            )
            index += 1

    pairs = candidate_pairs(left_records, right_records)
    occupancy_by_cell = bucket_occupancy(right_records)
    counts = tuple(occupancy_by_cell.values())
    left_index_by_id = {id(record): i for i, record in enumerate(left_records)}
    right_index_by_id = {id(record): i for i, record in enumerate(right_records)}
    comparison_pairs = tuple(
        (left_index_by_id[id(left)], right_index_by_id[id(right)])
        for left, right in pairs
    )
    return _PreparationSample(
        left_names=tuple(record.name for record in left_records),
        right_names=tuple(record.name for record in right_records),
        comparison_pairs=comparison_pairs,
        occupancy=BucketOccupancy(
            counts=counts,
            cell_size_degrees=DEFAULT_CELL_SIZE_DEGREES,
            # Named directly rather than left for a reader to reduce `counts`
            # themselves (BucketOccupancy's own docstring, Finding 1) — the
            # one number ADR-0009's claim ("the hoist pays in proportion to
            # re-comparison") is actually about.
            max_occupancy=max(counts),
        ),
    )


# A shared coordinate, not an invented geography: the only thing it needs to
# do is put both illustrative records in the same grid cell so
# candidate_pairs (ADR-0009, issue #65) has exactly one pair to score. Any
# equal pair of floats would do the same job.
_PREPARATION_COMPARISON_COORDINATE = (0.0, 0.0)


def _preparation_comparison(
    scorer: Scorer[Any], threshold: float, left_name: str, right_name: str
) -> PreparationComparison:
    """Both preparation paths' own score for this run's illustrative
    comparison (issue #65's third bullet: "the run record states which one
    produced each figure") — the same ``left_name``/``right_name`` pair
    every other per-arm measurement in this run already uses
    (:func:`_measure_arm`), scored by
    :func:`joinless.resolver.score_candidates` under each path in turn so
    neither is this run's default by accident (that function's own required
    ``preparation`` argument has no default either).

    The two records share one coordinate purely so
    :func:`~joinless.resolver.candidate_pairs`'s grid blocking pairs them
    with each other — the only candidate pair this comparison needs, and
    the same "one illustrative record" pattern :func:`_measure_arm` already
    uses for warm latency, peak memory and cold start.

    Cost is not measured here — issue #66's job, not this one's. This only
    proves, on this run's own real scorer and data, that the two paths
    agree — the general claim ``tests/test_resolver.py``'s property tests
    already established, exercised here for the one comparison this run
    actually makes.
    """
    latitude, longitude = _PREPARATION_COMPARISON_COORDINATE
    left = [
        Record(
            source="benchmark",
            ordinal=0,
            name=left_name,
            latitude=latitude,
            longitude=longitude,
        )
    ]
    right = [
        Record(
            source="benchmark",
            ordinal=0,
            name=right_name,
            latitude=latitude,
            longitude=longitude,
        )
    ]
    matcher: ThresholdMatcher[Any] = ThresholdMatcher(
        scorer=scorer, threshold=threshold
    )
    return PreparationComparison(
        hoisted=score_candidates(left, right, matcher, preparation="hoisted"),
        naive=score_candidates(left, right, matcher, preparation="naive"),
    )


def _measure_arm(
    arm: str,
    pooled: Corpus,
    left_name: str,
    right_name: str,
    power_mode: str,
    sample: _PreparationSample,
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

    ``sample`` is the run's one shared preparation-cost candidate set
    (issue #66, :func:`_build_preparation_sample`) — every arm's
    ``preparation_cost`` is timed over the same set, which is what makes
    the run's own :class:`~joinless.runrecord.BucketOccupancy` the occupancy
    that produced every arm's cost figure, not just one of them.
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
                preparation=Unavailable(arm=arm, reason=reason),
                preparation_cost=Unavailable(arm=arm, reason=reason),
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
    preparation = _preparation_comparison(scorer, frozen.value, left_name, right_name)
    preparation_cost = measure_preparation_cost(
        arm, sample.left_names, sample.right_names, sample.comparison_pairs
    )
    return (
        ArmResult(
            accuracy=accuracy,
            warm_latency=warm_latency,
            peak_memory=peak_memory,
            cold_start=cold_start,
            artifact_size=artifact_size,
            preparation=preparation,
            preparation_cost=preparation_cost,
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


# RFC-0001: "the fp32 and int8 arms are the same class with different model
# artefacts" — the pair this divergence compares is exactly this pair, named
# once here rather than re-spelled at each call site.
_INT8_DIVERGENCE_BASELINE_ARM = "embed-fp32"
_INT8_DIVERGENCE_CANDIDATE_ARM = "embed-int8"


def _int8_accuracy_divergence(
    arm_results: Mapping[str, ArmResult],
) -> Maybe[tuple[AccuracyDivergence, ...]]:
    """The int8 arm's per-family F1 divergence from the fp32 arm, computed
    from this run's own two accuracy reports (issue #67's third bullet) —
    never a second evaluation pass, and never asserted in prose: a reader who
    wants to know whether quantization changed a family's accuracy reads it
    off the run record itself, the same way :func:`_find_contradictions`
    already turns two arms' reports into a comparison rather than a claim.

    Requires both arms to have produced a real, comparable accuracy report in
    *this* run — an arm missing from ``arm_results`` entirely, or one whose
    ``accuracy`` is :class:`~joinless.evaluation.InvalidRun` (unavailable
    dependency, missing artefact, or any other reason :func:`_measure_arm`
    records), means there is nothing to compute a divergence from. That is
    reported as an explicit absence with a reason (ADR-0013) — never a
    fabricated comparison, and never an empty tuple, which would look like
    "computed, and nothing to report" rather than "not computed at all".
    """
    baseline_result = arm_results.get(_INT8_DIVERGENCE_BASELINE_ARM)
    if baseline_result is None or not isinstance(
        baseline_result.accuracy, EvaluationReport
    ):
        return Maybe(
            value=None,
            reason=(
                f"{_INT8_DIVERGENCE_BASELINE_ARM!r} did not produce a comparable "
                "accuracy report in this run"
            ),
        )
    candidate_result = arm_results.get(_INT8_DIVERGENCE_CANDIDATE_ARM)
    if candidate_result is None or not isinstance(
        candidate_result.accuracy, EvaluationReport
    ):
        return Maybe(
            value=None,
            reason=(
                f"{_INT8_DIVERGENCE_CANDIDATE_ARM!r} did not produce a comparable "
                "accuracy report in this run"
            ),
        )
    return Maybe(
        value=compute_accuracy_divergence(
            baseline=baseline_result.accuracy, candidate=candidate_result.accuracy
        ),
        reason=None,
    )


# ADR-0008 names two arm families, and ADR-0009 names the asymmetry between them
# as part of the hoist's finding: the classical arms recompute a cheap set or
# string operation on every naive re-preparation, the neural arms recompute an
# embedding — named once here rather than re-derived from each arm's own scorer
# class at call time.
_CLASSICAL_ARMS: frozenset[str] = frozenset({"overlap", "fuzzy"})
_NEURAL_ARMS: frozenset[str] = frozenset({"embed-fp32", "embed-int8"})


def _hoist_speedup(cost: PreparationCost) -> Metric:
    """How many times faster hoisted preparation was than naive, for one
    arm's own :class:`~joinless.measurement.PreparationCost` (issue #66's
    third bullet).

    Undefined, not infinite, when ``hoisted_seconds`` measured at ``0.0`` —
    a real possibility for the cheapest classical arms on a coarse clock —
    the same "undefined is not zero" rule ADR-0013 states everywhere else a
    figure in this run record might not be computable.
    """
    if cost.hoisted_seconds <= 0.0:
        return Metric(
            value=None,
            undefined_reason="hoisted preparation measured at 0 seconds",
        )
    return Metric(
        value=cost.naive_seconds / cost.hoisted_seconds, undefined_reason=None
    )


def _preparation_asymmetry(
    arm_results: Mapping[str, ArmResult],
    occupancy: BucketOccupancy,
) -> PreparationAsymmetry:
    """The classical/neural hoist speed-up comparison issue #66's third
    bullet asks to be "reported as a result, not as an aside" — computed
    once here, from every arm's own ``preparation_cost``, and handed to
    :meth:`~joinless.runrecord.RunAssembly.build` rather than left for a
    reader to compute by hand from the per-arm figures (:func:`_cmd_benchmark`).

    An arm whose ``preparation_cost`` is
    :class:`~joinless.measurement.Unavailable` — it never got far enough to
    be timed — contributes to neither mapping, mirroring how
    :func:`_find_contradictions` only ever compares arms with a real report
    to compare. An arm belonging to neither :data:`_CLASSICAL_ARMS` nor
    :data:`_NEURAL_ARMS` contributes to neither either — not reachable
    through :data:`_ARMS` today, since every registered arm is one or the
    other, but this function does not assume that of its argument.

    ``occupancy`` is the run's own :class:`~joinless.runrecord.BucketOccupancy`
    for the shared candidate set every arm's ``preparation_cost`` was
    measured over — carried on the returned
    :class:`~joinless.runrecord.PreparationAsymmetry` rather than left as a
    separate value the caller has to remember to attach later (Finding 1;
    see that type's own docstring for why occupancy belongs to the finding
    it explains).
    """
    classical: dict[str, Metric] = {}
    neural: dict[str, Metric] = {}
    for arm, result in arm_results.items():
        if not isinstance(result.preparation_cost, PreparationCost):
            continue
        speedup = _hoist_speedup(result.preparation_cost)
        if arm in _CLASSICAL_ARMS:
            classical[arm] = speedup
        elif arm in _NEURAL_ARMS:
            neural[arm] = speedup
    return PreparationAsymmetry(
        occupancy=occupancy,
        classical_speedups=MappingProxyType(classical),
        neural_speedups=MappingProxyType(neural),
    )


def _format_preparation_asymmetry(asymmetry: PreparationAsymmetry) -> list[str]:
    """The lines ``benchmark`` prints for its preparation-hoist asymmetry
    finding (issue #66's third bullet, mirroring :func:`_format_contradictions`'s
    own "findings... not a footnote" reasoning) — a pure function of the
    comparison :func:`_preparation_asymmetry` already computed, so what the
    command prints and what it persists in ``preparation_asymmetry`` can
    never say two different things about the same run.

    The occupancy this run's speed-ups were measured at is printed alongside
    them (Finding 1) — a reader of the printed output, not only of the JSON
    record, needs the same fact ADR-0009 requires travel with the result:
    "the hoist pays in proportion to re-comparison." The max is stated
    plainly rather than judged "small" or "large" here — this function does
    not invent a threshold; it hands the reader the one number ADR-0009's
    claim is about and the direction the claim predicts, and lets them judge
    it against whatever occupancy their own dataset has.
    """
    occupancy = asymmetry.occupancy
    lines = [
        "preparation hoist speed-up (naive seconds / hoisted seconds):",
        (
            f"  measured over candidate-bucket occupancy {list(occupancy.counts)} "
            f"(max {occupancy.max_occupancy}, cell size {occupancy.cell_size_degrees} "
            "degrees) — ADR-0009: the hoist's advantage grows with how full a "
            "bucket gets, so a denser dataset than this sample would show a larger "
            "speed-up, not this one"
        ),
    ]
    for label, speedups in (
        ("classical", asymmetry.classical_speedups),
        ("neural", asymmetry.neural_speedups),
    ):
        if not speedups:
            lines.append(f"  {label}: no arm in this run produced a comparable figure")
            continue
        for arm in sorted(speedups):
            metric = speedups[arm]
            value = "undefined" if metric.value is None else f"{metric.value:.2f}x"
            lines.append(f"  {label} {arm}: {value}")
    return lines


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
    # issue #66: one shared candidate set, drawn once, that every arm's
    # preparation cost is measured over - so this run's own bucket_occupancy
    # is the occupancy that produced every arm's cost figure, never a
    # parameter echoed back without the run that produced it (ADR-0009).
    preparation_sample = _build_preparation_sample(pooled)

    assembly = RunAssembly(expected_winners=_EXPECTED_WINNERS)
    arm_results: dict[str, ArmResult] = {}
    selected_thresholds: list[SelectedThreshold] = []
    # One entry per neural arm that actually initialised (issue #67): both
    # embed-fp32 and embed-int8 can load a model in the same run, each with
    # its own checksum, so this accumulates one identity per arm rather than
    # tracking "the last one that reported an identity" - a single slot for
    # that would have to silently drop one of the two.
    model_identities: dict[str, ModelIdentity] = {}
    for arm in _ARMS:
        result, selected, identity = _measure_arm(
            arm, pooled, left_name, right_name, power_mode, preparation_sample
        )
        assembly.add_arm(arm, result)
        arm_results[arm] = result
        if selected is not None:
            selected_thresholds.append(selected)
        if identity is not None:
            model_identities[arm] = identity

    # ADR-0011 rule 4 / issue #50: the pre-registered expectations are compared
    # against the actual outcome here, once, and the same value is both printed
    # below and persisted on the record - never recomputed for either.
    contradictions = _find_contradictions(_EXPECTED_WINNERS, arm_results)
    # issue #67's third bullet: computed once here, from this run's own two
    # accuracy reports, and persisted rather than recomputed - the same
    # discipline `contradictions` above already follows.
    int8_accuracy_divergence = _int8_accuracy_divergence(arm_results)
    # issue #66's third bullet: computed once here, from this run's own
    # per-arm preparation costs, and persisted rather than recomputed - the
    # same discipline `contradictions` and `int8_accuracy_divergence` above
    # already follow.
    preparation_asymmetry = _preparation_asymmetry(
        arm_results, preparation_sample.occupancy
    )

    # issue #68's third bullet: a graph that does not match its recorded
    # operator list must not produce a record at all - caught here, before
    # `assembly.build`/`write_record` runs, rather than folded into one arm's
    # own ArmResult the way a checksum mismatch is (ADR-0013 rule 3). Imported
    # lazily so a run that never reaches this branch never imports
    # joinless.embedding from here either (ADR-0014, ADR-0017) - though every
    # `benchmark` run already does, through embed-fp32/embed-int8's own probes
    # in the loop above.
    from joinless.embedding import QuantizedOperatorMismatchError

    try:
        quantized_operators = _quantized_operators(model_identities)
    except QuantizedOperatorMismatchError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    record = assembly.build(
        schema=_SCHEMA,
        started_at=started_at,
        command=("joinless", "benchmark"),
        environment=_environment(
            power_mode, models=model_identities, quantized_operators=quantized_operators
        ),
        evaluation_set=evaluation_set,
        selected_thresholds=tuple(selected_thresholds),
        contradictions=contradictions,
        int8_accuracy_divergence=int8_accuracy_divergence,
        preparation_asymmetry=preparation_asymmetry,
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
    for line in _format_preparation_asymmetry(preparation_asymmetry):
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
