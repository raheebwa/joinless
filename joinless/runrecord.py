# SPDX-License-Identifier: MIT
"""The run record: the durable artefact each benchmark run writes to
``benchmarks/`` (RFC-0002 "Output", benchmarks/README.md, issue #57).

This module owns exactly three things: the shape of a run record, the one
place it becomes JSON, and the one function that writes it to disk without
ever overwriting an earlier run. It builds no scorer, runs no evaluation, and
detects no hardware fact of its own — every value a caller wants recorded is
handed in already computed. That boundary is deliberate: :mod:`joinless.cli`
(the CLI wiring is the next agent's job) is where the real environment gets
read and the real command gets assembled; this module's job is only to make
what it is given into a record that means what it says.

It imports rather than redefines every type the modules that "just landed"
already own — :class:`~joinless.evaluation.Metric`,
:class:`~joinless.evaluation.EvaluationReport`,
:class:`~joinless.evaluation.SelectedThreshold`,
:class:`~joinless.evaluation.FrozenThreshold`,
:class:`~joinless.evaluation.ExpectedWinners`,
:class:`~joinless.evaluation.Contradiction`,
:class:`~joinless.measurement.Unavailable`,
:class:`~joinless.measurement.WarmLatency`,
:class:`~joinless.measurement.PeakMemory` and
:class:`~joinless.measurement.ColdStartPhases` — so a run record is built
from the same values a caller already has, not a second, parallel
representation of them.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Generic, TypeVar, cast

from joinless.corpus import Corpus
from joinless.evaluation import (
    AccuracyDivergence,
    Contradiction,
    ExpectedWinners,
    InvalidRun,
    Metric,
    SealedTestAccuracy,
    SelectedThreshold,
)
from joinless.measurement import (
    ColdStartPhases,
    PeakMemory,
    PreparationCost,
    Unavailable,
    WarmLatency,
)
from joinless.resolver import PreparationPath, ScoredComparisons

_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class Maybe(Generic[_T]):
    """A metadata field that may not apply to this run.

    This is the run record's version of the same rule
    :class:`joinless.evaluation.Metric` states for a number: ``value`` is
    ``None`` exactly when ``reason`` is set, so a field that cannot be
    determined — a model checksum when no arm in this run loads a model, a
    quantized-operator list for a classical-only run — is written as
    ``null`` with a reason, never as an absent key, a zero, or an invented
    value (ADR-0013's fail-closed rule, applied here to metadata rather than
    a measured figure). ``Maybe`` is generic over the value's type because
    metadata fields are not all numbers the way a :class:`Metric` is; reusing
    ``Metric`` itself for a string would type-check against nothing and
    would still be a second copy of the same invariant.
    """

    value: _T | None
    reason: str | None

    def __post_init__(self) -> None:
        if (self.value is None) != (self.reason is not None):
            raise ValueError(
                "Maybe.value must be None exactly when reason is set "
                f"(got value={self.value!r}, reason={self.reason!r})"
            )


@dataclass(frozen=True, slots=True)
class Hardware:
    """CPU, core count and memory (benchmarks/README.md's first bullet), plus
    ``python_version``, ``system`` and ``release`` — the exact field set the
    one existing record in ``benchmarks/`` already carries under
    ``environment.hardware``. New fields for this schema live on
    :class:`Environment` instead of here, so this type's shape stays the one
    already in use rather than a second one invented alongside it.
    """

    cpu_count: int
    machine: str
    python_version: str
    release: str
    system: str
    total_memory_bytes: int


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    """The model identity, revision, checksum and licence RFC-0002 Method
    step 5 requires (benchmarks/README.md: "Model identity, revision and
    checksum, where applicable") and issue #59's fourth bullet requires
    alongside them ("the model card's licence is recorded alongside the
    artefact identity"). Wrapped in a :class:`Maybe` on :class:`Environment`
    rather than making each field its own ``Maybe`` — when no arm in a run
    loads a model, all four are inapplicable for the same one reason, not
    four independently missing values.
    """

    model_id: str
    revision: str
    checksum_sha256: str
    license: str


@dataclass(frozen=True, slots=True)
class MatmulConversion:
    """How many of one candidate-for-quantization operator type converted, and
    how many remain fp32 (issue #68: "the list is what makes the int8 number
    interpretable" — a bare list of the two replacement operator types present
    cannot answer "36 matmuls converted or 3", only whether conversion
    happened at all). Mirrors
    ``benchmarks/20260812T181752Z-quantization-spike.json``'s own
    ``operators.matmul_conversion`` shape, the precedent this run record's own
    shape is read against (:mod:`joinless.embedding`'s
    :func:`~joinless.embedding.matmul_conversion_census`).

    ``fp32_count`` is redundant with ``converted_count + int8_count_remaining``
    — kept anyway, mirroring the spike record's own three-field shape exactly,
    so a reader comparing this run's numbers against that record's finds the
    same three names rather than having to recompute one of them.
    """

    converted_count: int
    fp32_count: int
    int8_count_remaining: int


@dataclass(frozen=True, slots=True)
class RuntimeVersions:
    """ONNX Runtime and ``rapidfuzz`` versions (benchmarks/README.md).
    ``rapidfuzz`` is never absent — it is a base dependency of every install
    profile (ADR-0014) — so it carries a plain ``str``. ``onnxruntime`` is
    installed only under the ``neural`` profile, so it is a :class:`Maybe`,
    null with a reason on a run that never constructs a neural arm (deferred
    from issue #34 to this module, as its own deferral comment records).
    """

    onnxruntime: Maybe[str]
    rapidfuzz: str


@dataclass(frozen=True, slots=True)
class Environment:
    """Everything RFC-0002 Method step 5 requires recorded about the machine
    and configuration a run executed under, gathered into the one place
    benchmarks/README.md calls "every record carries".

    Every field here is supplied by the caller — this module reads no
    hardware, clock, or filesystem fact of its own (module docstring), so a
    test can build one from plain values without patching anything.

    ``models`` carries one :class:`ModelIdentity` per neural arm that
    actually initialised, keyed by arm name — a plain mapping rather than a
    single :class:`Maybe`, because a run can load more than one neural arm's
    model at once (``embed-fp32`` and ``embed-int8``, issue #67) and each
    carries its own checksum: a singular slot would have to pick one and
    silently drop the other, which is exactly the quiet degradation ADR-0013
    exists to rule out. An empty mapping means no neural arm ran, unambiguously
    — there is no zero/undefined confusion to guard against here the way
    there is for a ratio (ADR-0013's motivating case), since "which arms
    loaded a model" has only one honest empty state.

    ``quantized_operators`` is the int8 arm's matmul-conversion census, keyed
    by candidate operator type (``"MatMul"``, ``"Gemm"``) — issue #68's stated
    purpose is answering "how many of the graph's matmuls were converted and
    how many remain in fp32", which a flat list of the operator types present
    cannot do: a smaller artefact and an unchanged latency are either the
    expected consequence of a partly-quantized encoder or a puzzling result,
    and only the counts tell a reader which. Wrapped in :class:`Maybe`, not a
    bare mapping — a run with no int8 arm has no graph to read a census from
    (ADR-0013).

    ``measurement_preparation_path`` states which of
    :data:`~joinless.resolver.PreparationPath`'s two call patterns produced
    every arm's ``results.<arm>.accuracy`` and ``results.<arm>.warm_latency``
    in this run (Finding 2; issue #65's third bullet: "the run record states
    which one produced each figure") — always ``"hoisted"``:
    :func:`joinless.evaluation.evaluate_sealed_test` prepares every pair's
    names in one ``prepare_all`` batch ahead of scoring (that module's own
    docstring), and this project's warm-latency worker prepares each side
    once before its repeated ``score`` calls; neither has a naive
    counterpart a run could select instead, so the value is a structural
    fact about those two call sites, identical for every arm in every run,
    not a per-arm choice.

    One field here rather than a new field added to
    :class:`~joinless.evaluation.EvaluationReport` and
    :class:`~joinless.measurement.WarmLatency` themselves: both types are
    pure figures owned by modules that know nothing about preparation paths
    at all (:mod:`joinless.evaluation`'s own docstring: "nothing reads a
    file, spawns a process..."), and a value that never varies does not
    earn a new field on every construction site of two types this project
    already reuses across many unrelated tests (YAGNI) — recording it once,
    here, where every other run-wide measurement-methodology fact
    (``warmup_count``, ``repetition_count``, ``thread_count``) already
    lives, says the same thing without it.

    ``peak_memory``, ``cold_start`` and ``artifact_size`` are not covered
    by this field: each measures exactly one comparison with nothing to
    hoist across, so "hoisted" and "naive" collapse to the same operation
    for them and a path label would state a fact that is always trivially
    true. ``preparation`` and ``preparation_cost`` are not covered either —
    both already exercise, and self-tag, both paths in every run
    (:class:`PreparationComparison`, :class:`~joinless.measurement.PreparationCost`),
    so a third label naming which one produced them would have nothing to
    disambiguate.
    """

    hardware: Hardware
    runtime_versions: RuntimeVersions
    power_mode: str
    thread_count: int
    warmup_count: int
    repetition_count: int
    models: Mapping[str, ModelIdentity]
    quantized_operators: Maybe[Mapping[str, MatmulConversion]]
    measurement_preparation_path: PreparationPath


@dataclass(frozen=True, slots=True)
class EvaluationSetIdentity:
    """Which corpora produced this run's numbers, and their composition
    (benchmarks/README.md: "Evaluation set identity and case mixture").

    ``seeds`` is what lets a reader regenerate the exact same corpora via
    :func:`joinless.corpus.generate_corpus` — a corpus is a pure function of
    its seed (that module's docstring), so the seed *is* the identity.
    ``case_mixture`` is the pair count per family, pooled across every
    corpus this run drew from: the composition ADR-0011 rule 4 requires
    visible, rather than left to be inferred from the constants in
    :mod:`joinless.corpus`.
    """

    seeds: tuple[int, ...]
    case_mixture: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class PreparationComparison:
    """This run's naive and hoisted preparation paths, run over the same
    illustrative comparison and each self-tagged with the path that
    produced it (issue #65's third bullet: "the run record states which
    one produced each figure").

    Holding one :class:`~joinless.resolver.ScoredComparisons` per path —
    rather than a bare pair of floats — is what makes that attribution
    structural: :attr:`~joinless.resolver.ScoredComparisons.path` travels
    on the value itself (that type's own docstring), so a reader holding
    ``comparison.hoisted`` never has to trust this field's name, or its
    position in this dataclass, to know which call pattern produced it.

    Cost is deliberately absent here — issue #66's job, not this type's.
    Both paths are exercised in a real run to prove they agree on this
    run's own data (ADR-0009 consequence 2: the naive path is a control,
    not a stub), never to time either one.
    """

    hoisted: ScoredComparisons
    naive: ScoredComparisons


@dataclass(frozen=True, slots=True)
class BucketOccupancy:
    """The candidate-bucket occupancy distribution for the record set this
    run's preparation-cost measurement drew comparisons from (issue #66) —
    the independent variable the hoist's win is a function of, ADR-0009
    says, not context to mention alongside it.

    ``counts`` is one entry per occupied grid cell on the candidate side —
    the raw distribution, not reduced to a mean: issue #66 asks for "a
    distribution, not a mean", and a mean here would already be the wrong
    summary before a reader even sees it, since ADR-0009's claim is about
    the *shape* of occupancy, not its average. A reader who wants a mean,
    median or histogram computes it from this tuple; nothing this module
    could compute instead would be more informative than the counts
    themselves. There is exactly one ``BucketOccupancy`` per run, not one
    per arm — occupancy is a property of the candidate set every arm's
    preparation cost is measured over, not of any one arm's scorer.

    ``cell_size_degrees`` is the grid cell width the distribution was
    measured under (:data:`joinless.resolver.DEFAULT_CELL_SIZE_DEGREES`
    unless a caller overrides it) — the counts mean nothing without knowing
    how wide a cell is.

    ``max_occupancy`` is redundant with ``max(counts)`` — kept anyway, the
    same "computed once, named directly" precedent :class:`MatmulConversion`'s
    own ``fp32_count`` already sets for a value a reader would otherwise have
    to recompute: ADR-0009's claim is specifically that the hoist's win grows
    with how full the fullest bucket gets, so the one number that claim is
    about should not require reducing a tuple to find (Finding 1).
    """

    counts: tuple[int, ...]
    cell_size_degrees: float
    max_occupancy: int


@dataclass(frozen=True, slots=True)
class PreparationAsymmetry:
    """The classical/neural asymmetry ADR-0009 names as itself part of the
    finding (issue #66's third bullet: "the asymmetry between classical and
    neural arms is reported as a result, not as an aside") — each arm's
    hoist speed-up (naive preparation seconds divided by hoisted), grouped
    by arm family so a reader sees the two groups directly rather than
    first having to know which of the four registered arms are classical
    and which are neural before partitioning four numbers by hand.

    Keyed by arm name within each group, mapping to a
    :class:`~joinless.evaluation.Metric` rather than a bare ``float``: a
    hoisted preparation timed at ``0.0`` seconds — a real possibility for
    the cheapest classical arms on a coarse clock — would make the ratio
    undefined, not infinite, the same "undefined is not zero" rule
    ADR-0013 states everywhere else this module reports a figure. An arm
    that never produced a comparable :class:`~joinless.measurement.PreparationCost`
    in this run is absent from both mappings entirely, mirroring how
    :func:`~joinless.evaluation.find_contradictions` only ever compares
    arms with a real report to compare.

    ``occupancy`` is the candidate-bucket occupancy distribution the run's
    shared preparation-cost sample was drawn from (:class:`BucketOccupancy`,
    issue #66) — carried here, on the finding it explains, rather than as an
    unrelated top-level field on :class:`RunRecord` beside ``evaluation_set``
    (Finding 1). This run's accuracy evaluation performs no grid blocking at
    all — it scores each of ``evaluation_set``'s labelled pairs directly — so
    an occupancy sitting next to ``evaluation_set`` invited exactly the
    misreading ADR-0009 exists to prevent: that the accuracy figures came
    from a run whose candidate buckets held 1 to 4 records, when in truth
    this occupancy describes only the one small sample every arm's own
    :class:`~joinless.measurement.PreparationCost` was measured over.
    Nesting it inside the speed-up comparison it explains is what makes that
    scope structural rather than a fact a reader has to already know —
    ADR-0009's own words: "a hoist speed-up quoted without the occupancy
    that produced it cannot be transferred to any other dataset."
    """

    occupancy: BucketOccupancy
    classical_speedups: Mapping[str, Metric]
    neural_speedups: Mapping[str, Metric]


@dataclass(frozen=True, slots=True)
class ArmResult:
    """Everything one arm contributed to a run: its accuracy report and its
    four resource measurements (RFC-0002 Metrics table), each either its
    measured value or an ADR-0013 outcome explaining why not —
    :class:`~joinless.evaluation.InvalidRun` for accuracy,
    :class:`~joinless.measurement.Unavailable` for a measurement — never
    omitted and never coerced into a number.

    ``accuracy`` is typed :class:`~joinless.evaluation.SealedTestAccuracy`,
    not a bare :class:`~joinless.evaluation.EvaluationReport` (issue #97):
    the pooled per-family figure travels with the per-seed reports and the
    seed-to-seed variation behind it, and :class:`SealedTestAccuracy`'s own
    construction refuses a pooled figure reported without that variation
    (ADR-0011 rule 3) — a run whose threshold selection was itself
    contaminated still reports :class:`~joinless.evaluation.InvalidRun`
    here, exactly as before.

    ``artifact_size`` is typed :class:`~joinless.evaluation.Metric`, not one
    of the three dedicated measurement dataclasses its siblings use: unlike
    peak memory or cold start, this figure carries no extra fields beyond a
    number and why there might not be one, which is exactly what ``Metric``
    already is (issue #63) — a second, near-identical dataclass would be a
    third shape for the same "a value that may be undefined" idea
    :class:`~joinless.evaluation.Metric` and :class:`Maybe` already cover
    between them. ``Unavailable`` is still the outer alternative, for an arm
    that never got far enough to be asked: the classical arms *do*
    initialise and their ``Metric`` is defined-undefined (a real "no
    artefact" fact); an arm whose ``get_scorer`` call itself failed never
    reaches that question at all.

    ``preparation`` is a fifth figure, alongside the four RFC-0002 names —
    this arm's naive and hoisted preparation paths, run over the same
    candidate pair and each self-tagged with the path that produced it
    (issue #65's third bullet, :class:`PreparationComparison`). Required,
    with no default, for the same reason ``resolver.score_candidates``'s
    own ``preparation`` argument has none: neither path is this run's
    default by accident. ``Unavailable`` for the same "never got far
    enough to be asked" arms as its siblings.

    ``preparation_cost`` is a sixth figure, alongside ``preparation`` —
    this arm's own hoisted and naive preparation *cost*, timed in an
    isolated worker over the run's shared occupancy sample (issue #66,
    :class:`~joinless.measurement.PreparationCost`), as distinct from
    ``preparation``'s equality proof: that field asks "do the two paths
    agree on this run's data", never timed; this one asks "what does each
    path cost", never scored. Both exist on every arm because they answer
    different questions issue #65 and issue #66 each ask, not because one
    supersedes the other.
    """

    accuracy: SealedTestAccuracy | InvalidRun
    warm_latency: WarmLatency | Unavailable
    peak_memory: PeakMemory | Unavailable
    cold_start: ColdStartPhases | Unavailable
    artifact_size: Metric | Unavailable
    preparation: PreparationComparison | Unavailable
    preparation_cost: PreparationCost | Unavailable


@dataclass(frozen=True, slots=True)
class RunRecord:
    """One benchmark run, exactly as RFC-0002's "Output" section and
    benchmarks/README.md describe it. The only way to obtain one is
    :meth:`RunAssembly.build` — see that class for why.

    ``contradictions`` is every family whose actual best-scoring arm, among
    ``results``, was not the one pre-registered in ``expected_winners`` — the
    output of :func:`joinless.evaluation.find_contradictions`, computed once
    by the caller and handed to :meth:`RunAssembly.build` rather than
    recomputed here (issue #50: "a contradiction found at run time and
    recomputed at render time is two answers that can disagree"). An empty
    tuple means the comparison ran and nothing broke — a genuine, positive
    outcome — never "the comparison never happened": the field is a required
    argument to :meth:`RunAssembly.build`, so no ``RunRecord`` can exist
    without it having been supplied.

    ``int8_accuracy_divergence`` is the int8 arm's per-family F1 divergence
    from the fp32 arm, in the same run (issue #67's third bullet) — the
    output of :func:`joinless.evaluation.compute_accuracy_divergence`,
    computed once by the caller from the two arms' own
    :class:`~joinless.evaluation.EvaluationReport` and handed to
    :meth:`RunAssembly.build`, mirroring exactly how ``contradictions`` is
    computed once and persisted rather than recomputed. Wrapped in
    :class:`Maybe`, not a bare tuple: a run where either arm did not produce
    a comparable accuracy report — the int8 artefact absent, the fp32 arm
    unavailable — has nothing to compute a divergence from, and that is
    reported as an explicit absence with a reason (ADR-0013), never as an
    empty tuple that would look like "computed, and there was nothing to
    report" rather than "not computed at all". Also a required argument to
    :meth:`RunAssembly.build`, for the same reason ``contradictions`` is: a
    caller cannot build a record without stating what it found.

    ``preparation_asymmetry`` is the classical/neural hoist speed-up
    comparison issue #66's third bullet asks to be "reported as a result,
    not as an aside" (:class:`PreparationAsymmetry`) — computed once by the
    caller from every arm's own :class:`~joinless.measurement.PreparationCost`
    and handed to :meth:`RunAssembly.build`, the same discipline
    ``contradictions`` and ``int8_accuracy_divergence`` already follow:
    computed once, persisted, never recomputed at render time. It also
    carries the run's :class:`BucketOccupancy` (Finding 1) — there is no
    separate ``bucket_occupancy`` field on this type for the same reason
    there is no separate one on :class:`PreparationAsymmetry` itself: see
    that type's own docstring for why occupancy belongs to the finding it
    explains rather than to this type directly.
    """

    schema: str
    record_id: str
    started_at: datetime
    command: tuple[str, ...]
    environment: Environment
    evaluation_set: EvaluationSetIdentity
    selected_thresholds: tuple[SelectedThreshold, ...]
    expected_winners: ExpectedWinners
    results: Mapping[str, ArmResult]
    contradictions: tuple[Contradiction, ...]
    int8_accuracy_divergence: Maybe[tuple[AccuracyDivergence, ...]]
    preparation_asymmetry: PreparationAsymmetry


def build_evaluation_set_identity(
    corpora: Sequence[Corpus],
) -> EvaluationSetIdentity:
    """The seeds and pooled per-family pair counts for every corpus in
    ``corpora`` — grounded in the real, generated corpora a run actually
    drew from (benchmarks/README.md, RFC-0002 Method step 5), rather than
    hand-written to match :mod:`joinless.corpus`'s constants, which could
    drift out of sync with what a run really used.
    """
    if not corpora:
        raise ValueError("evaluation set identity requires at least one corpus")
    case_mixture: dict[str, int] = {}
    for corpus in corpora:
        for pair in corpus.pairs:
            if pair.category is None:
                raise ValueError(
                    f"pair {pair.pair_id!r} has no category; case mixture requires one"
                )
            case_mixture[pair.category] = case_mixture.get(pair.category, 0) + 1
    return EvaluationSetIdentity(
        seeds=tuple(corpus.seed for corpus in corpora),
        case_mixture=MappingProxyType(case_mixture),
    )


_SLUG = "benchmark"


def build_record_id(started_at: datetime) -> str:
    """The stable identifier issue #57 asks for, and the naming convention
    it says is unsettled anywhere else: ``<basic ISO 8601 UTC
    timestamp>-benchmark.json`` — the same shape already in use for
    ``benchmarks/20260812T181752Z-quantization-spike.json``, generalised
    with a slug fixed to this schema rather than the spike's own.

    This *is* the file name :func:`write_record` writes to, not a value
    computed independently of it — a bug report naming this identifier
    names the exact file, with nothing to keep in sync between the two.
    """
    if started_at.tzinfo is None:
        raise ValueError("started_at must be timezone-aware")
    timestamp = started_at.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{_SLUG}.json"


class RunAssembly:
    """The one way to build a :class:`RunRecord` — and the mechanism issue
    #50 asks for: "a run that has not been given its expectations before its
    reports should not be constructible."

    ``expected_winners`` is a required constructor argument, so an assembly
    cannot exist without one. Every arm's result is added afterward, through
    :meth:`add_arm`, which exists only on an already-constructed assembly —
    there is no function in this module that accepts an arm's result without
    first requiring an assembly to call it on, and no way to obtain that
    assembly except by supplying expectations. The ordering issue #50 wants
    enforced is therefore a fact about which objects exist and which methods
    are reachable from them, the same device
    :class:`~joinless.evaluation.FrozenThreshold` uses to gate
    :func:`~joinless.evaluation.evaluate_sealed_test` — not a runtime check
    for the actual chronological order of two calls, which no in-process API
    can observe.
    """

    def __init__(self, expected_winners: ExpectedWinners) -> None:
        self._expected_winners = expected_winners
        self._results: dict[str, ArmResult] = {}

    def add_arm(self, arm: str, result: ArmResult) -> None:
        """Attach ``arm``'s result. Calling this twice for the same arm
        replaces its result rather than raising — a run record is assembled
        once, by one caller, in one place; there is no scenario in which two
        different results for the same arm in the same run are both worth
        keeping, so the second write is treated as a correction rather than
        an error."""
        self._results[arm] = result

    def build(
        self,
        *,
        schema: str,
        started_at: datetime,
        command: Sequence[str],
        environment: Environment,
        evaluation_set: EvaluationSetIdentity,
        selected_thresholds: Sequence[SelectedThreshold],
        contradictions: Sequence[Contradiction],
        int8_accuracy_divergence: Maybe[tuple[AccuracyDivergence, ...]],
        preparation_asymmetry: PreparationAsymmetry,
    ) -> RunRecord:
        """Freeze everything added so far into a :class:`RunRecord`.

        Raises if no arm was ever added: a run record with an empty
        ``results`` mapping would look like a sound zero-arm comparison
        rather than the caller error it is — the same reasoning
        :func:`joinless.evaluation.evaluate` applies to an empty pair list.

        ``contradictions`` and ``int8_accuracy_divergence`` have no default:
        a caller cannot build a record without stating what
        :func:`joinless.evaluation.find_contradictions` and
        :func:`joinless.evaluation.compute_accuracy_divergence` found, which
        is what makes "the comparison ran, whatever it found" a fact about
        every ``RunRecord`` rather than a step a caller could forget (see
        :class:`RunRecord`'s own docstring). ``preparation_asymmetry`` has
        no default for the same reason (issue #66) — and it is the only
        place :class:`BucketOccupancy` enters a record at all (Finding 1),
        so there is no second, separate occupancy argument here either.
        """
        if not self._results:
            raise ValueError("run record requires at least one arm's result")
        return RunRecord(
            schema=schema,
            record_id=build_record_id(started_at),
            started_at=started_at,
            command=tuple(command),
            environment=environment,
            evaluation_set=evaluation_set,
            selected_thresholds=tuple(selected_thresholds),
            expected_winners=self._expected_winners,
            results=MappingProxyType(dict(self._results)),
            contradictions=tuple(contradictions),
            int8_accuracy_divergence=int8_accuracy_divergence,
            preparation_asymmetry=preparation_asymmetry,
        )


# ADR-0013 point 2: "a reader who never saw the run cannot mistake it for a sound
# one." Each of these types only ever appears as one half of an ADR-0013 either/or
# slot on ArmResult, so tagging by concrete type — rather than asking every branch
# to carry its own "am I the failure case" flag — is unambiguous and exhaustive.
_FAILURE_STATUS_BY_TYPE: dict[type, str] = {
    InvalidRun: "invalid",
    Unavailable: "unavailable",
}
_OK_TAGGED_TYPES: tuple[type, ...] = (
    SealedTestAccuracy,
    WarmLatency,
    PeakMemory,
    ColdStartPhases,
    PreparationComparison,
    PreparationCost,
)


def _dataclass_dict(value: Any) -> dict[str, Any]:
    return {f.name: _to_jsonable(getattr(value, f.name)) for f in fields(value)}


def _to_jsonable(value: Any) -> Any:
    """The one place a :class:`RunRecord` — or any value nested inside one —
    becomes something :mod:`json` can write (module docstring: "keep
    serialisation in one place so a field cannot be written one way here and
    another way there"). Recursive and generic over every dataclass this
    module and its neighbours define, so a new field on any of them is
    written correctly without a second function learning about it.
    """
    failure_status = _FAILURE_STATUS_BY_TYPE.get(type(value))
    if failure_status is not None:
        return {"status": failure_status, **_dataclass_dict(value)}
    if isinstance(value, _OK_TAGGED_TYPES):
        return {"status": "ok", **_dataclass_dict(value)}
    if is_dataclass(value) and not isinstance(value, type):
        return _dataclass_dict(value)
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (frozenset, set)):
        return sorted(_to_jsonable(item) for item in value)
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def record_to_dict(record: RunRecord) -> dict[str, Any]:
    """``record``, as a plain ``dict`` of JSON-serialisable values — the only
    function this module or any caller uses to turn a :class:`RunRecord`
    into data, so a field cannot be written one way by one caller and
    another way by a second (module docstring)."""
    return cast(dict[str, Any], _to_jsonable(record))


def write_record(record: RunRecord, directory: Path) -> Path:
    """Write ``record`` to ``directory`` under its own
    :attr:`RunRecord.record_id`, and return the path written.

    Never overwrites (issue #57; benchmarks/README.md: "Records are never
    edited after the fact... a later run is a new record, not an
    overwrite"): the file is opened in exclusive-creation mode (``"x"``),
    which fails atomically if the name is already taken, rather than a
    separate ``path.exists()`` check followed by a write — two steps that
    are not atomic against each other and would leave a window in which a
    second run could win a race the caller never sees. ``directory`` is a
    required parameter rather than a hard-coded ``benchmarks/`` path, which
    is what lets a test point it at a temporary directory instead of the
    real one (module docstring: this module reads no filesystem fact of its
    own beyond what it is told to write to).
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / record.record_id
    payload = json.dumps(record_to_dict(record), indent=2, sort_keys=True) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
    return path
