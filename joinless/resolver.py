# SPDX-License-Identifier: MIT
"""The resolver: identity, candidate generation, scoring, tie-break, merge.

FR-1 — two record sets with no shared key, resolved into one merged set — is the
composition this module wires together, and every stage but one is already built
elsewhere: identity is :mod:`joinless.records`, similarity and thresholding are the
seam in :mod:`joinless.scoring` (RFC-0001). What is new here is candidate generation,
the distance tie-break, the merge policy, and :func:`resolve`, which composes all of
it into one entry point.

**Candidate generation (FR-2).** Comparing every left record against every right
record is O(n*m). A coordinate grid replaces that with a comparison against a
record's own cell and its eight neighbours: :func:`bucket_occupancy` indexes the
coordinate-bearing right records by cell, and :func:`candidate_pairs` (and
:func:`resolve` internally) looks each left record up in that index. The complexity
claim is conditional and the condition matters: comparison count is **linear in
record count under bounded bucket occupancy**, because a record is compared against
every candidate in nine cells — and it degrades toward quadratic as occupancy grows,
because there is no longer a bound on how many candidates live in those nine cells.
That is why :func:`bucket_occupancy` is a public function rather than an internal
detail: the claim cannot be trusted without a way to check the condition it depends
on against real data.

**Coordinate-less records (FR-3).** A record without coordinates cannot be placed in
a grid cell, so it can never appear as a candidate for, or of, anything — no scorer,
classical or neural, changes that, because the seam only ever sees records that
candidate generation already selected. Such a record is retained in the output as an
unmatched source record with a reason, never silently dropped. It is retained, not
linked: FR-3 draws that line deliberately, and closing it would need a name-only
blocking strategy, which is out of scope for this project.

**Matching is per left record, not a global assignment.** For each left record with
coordinates, :func:`resolve` picks the single best-scoring candidate that clears the
matcher's threshold, breaking a tied score by distance (below). It does not run a
global one-to-one assignment across the whole batch. Two left records that both sit
in one grid cell can legitimately name the same right record — for instance, two
listings for units at one address matching a single anchor record for that address —
so a right record being claimed by more than one left record is a real outcome this
resolver allows rather than an assignment conflict to break. No requirement here asks
for exclusivity on the right side, and enforcing it would add a global assignment
step that answers a question nobody asked.

**The distance tie-break belongs to the resolver, not the scorer** (RFC-0001).
Candidates within one left record's nine-cell neighbourhood can tie on name score;
when that happens the closer candidate, by great-circle distance, wins. That keeps a
scorer swap unable to touch anything about *how* a decision among tied candidates is
reached — only which candidates score high enough to be tied in the first place.

**Merge policy (FR-5).** :func:`merge` is a pure function of two records, decoupled
from candidate generation on purpose: the "more-populated record wins" rule needs a
record that has no coordinates but many other fields, and the coordinate override
needs the opposite, and neither combination is one grid bucketing ever produces (a
matched pair always has coordinates on both sides). Testing the three rules directly
against :func:`merge`, rather than only through :func:`resolve`, is what makes each
one an independently falsifiable claim instead of an accident of what candidate
generation happened to feed it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

from joinless.records import Record, record_id
from joinless.scoring import ThresholdMatcher

# A city-scale default: about 1.1 km of latitude at the equator, and less of
# longitude away from it. Public-record and civic listings (PRD §1) are typically
# geocoded to this precision or coarser, so the default groups plausible neighbours
# into shared cells without being so wide that unrelated listings collide. A caller
# resolving a different kind of dataset passes its own value.
DEFAULT_CELL_SIZE_DEGREES = 0.01

_NEIGHBOUR_OFFSETS: tuple[tuple[int, int], ...] = tuple(
    (row_delta, col_delta) for row_delta in (-1, 0, 1) for col_delta in (-1, 0, 1)
)

_EARTH_RADIUS_KM = 6371.0088

_REASON_NO_COORDINATES = (
    "no coordinates: a record without them can never enter a candidate set (FR-3)"
)
_REASON_NO_CANDIDATES = "no candidate record shares its nine-cell grid neighbourhood"
_REASON_NO_MATCH_ABOVE_THRESHOLD = "no candidate's score met the matcher's threshold"
_REASON_NOT_SELECTED = "not chosen as any left record's best-scoring candidate"


@dataclass(frozen=True, slots=True)
class MergedRecord:
    """The content of one matched pair, merged under the FR-5 policy.

    Not a :class:`~joinless.records.Record`: a merged entity spans two sources, and
    ``Record`` deliberately carries exactly one. ``sources`` is the provenance FR-5
    asks for — every source that contributed to this merge, not which of its two
    records won which field.
    """

    name: str
    latitude: float | None
    longitude: float | None
    fields: Mapping[str, str]
    sources: frozenset[str]


@dataclass(frozen=True, slots=True)
class MatchedPair:
    """One resolved link between a left and a right record, and their merge."""

    left: Record
    right: Record
    merged: MergedRecord


@dataclass(frozen=True, slots=True)
class UnmatchedRecord:
    """A source record with no place in any matched pair, and why."""

    record: Record
    reason: str


@dataclass(frozen=True, slots=True)
class ResolutionResult:
    """FR-1's "one merged set", reported as the parts a caller needs to audit it:
    the matched pairs, the unmatched records with their reasons, and the
    candidate-generation occupancy those pairs were drawn from (FR-2)."""

    pairs: tuple[MatchedPair, ...]
    unmatched: tuple[UnmatchedRecord, ...]
    occupancy: Mapping[tuple[int, int], int]

    @property
    def merged(self) -> tuple[Record | MergedRecord, ...]:
        """The one merged set FR-1 asks for: each pair's merge, plus every
        unmatched record retained exactly as it was (FR-3)."""
        return tuple(pair.merged for pair in self.pairs) + tuple(
            entry.record for entry in self.unmatched
        )


def _has_coordinates(record: Record) -> bool:
    """``latitude`` and ``longitude`` are independent ``Optional`` fields on
    :class:`Record` — a row carrying only one of the two is exactly as
    unplaceable in a grid cell as one carrying neither. Used wherever a
    caller only needs the yes/no question; :func:`_cell` repeats the
    two-field test itself because it also needs the narrowed values, which
    a separate boolean function cannot hand back to it.
    """
    return record.latitude is not None and record.longitude is not None


def _cell(record: Record, cell_size: float) -> tuple[int, int] | None:
    """The grid cell a record's coordinates fall in, or ``None`` without them."""
    if record.latitude is None or record.longitude is None:
        return None
    return (
        math.floor(record.latitude / cell_size),
        math.floor(record.longitude / cell_size),
    )


def _index_by_cell(
    records: Sequence[Record], cell_size: float
) -> dict[tuple[int, int], list[Record]]:
    """Coordinate-bearing records grouped by grid cell. Coordinate-less records are
    excluded here, not filtered by a caller, so every consumer of this index —
    candidate lookup and occupancy alike — agrees on what counts as indexed."""
    index: dict[tuple[int, int], list[Record]] = {}
    for record in records:
        cell = _cell(record, cell_size)
        if cell is None:
            continue
        index.setdefault(cell, []).append(record)
    return index


def _candidates_for(
    record: Record,
    index: Mapping[tuple[int, int], Sequence[Record]],
    cell_size: float,
) -> list[Record]:
    """Every record in ``index`` sharing ``record``'s cell or one of its eight
    neighbours. Empty if ``record`` has no coordinates or no cell is populated."""
    cell = _cell(record, cell_size)
    if cell is None:
        return []
    row, col = cell
    return [
        candidate
        for row_delta, col_delta in _NEIGHBOUR_OFFSETS
        for candidate in index.get((row + row_delta, col + col_delta), ())
    ]


def bucket_occupancy(
    records: Sequence[Record], cell_size: float = DEFAULT_CELL_SIZE_DEGREES
) -> Mapping[tuple[int, int], int]:
    """Count of coordinate-bearing records in each grid cell.

    Exposed on its own rather than left as an internal detail of candidate
    generation (issue #26): the module docstring's complexity claim — linear under
    bounded occupancy, degrading toward quadratic as occupancy grows — is
    conditional, and a caller needs a way to check that condition against its own
    data rather than take the claim on trust.
    """
    return {
        cell: len(bucket) for cell, bucket in _index_by_cell(records, cell_size).items()
    }


def candidate_pairs(
    left: Sequence[Record],
    right: Sequence[Record],
    cell_size: float = DEFAULT_CELL_SIZE_DEGREES,
) -> list[tuple[Record, Record]]:
    """Every ``(left, right)`` pair sharing a nine-cell grid neighbourhood (FR-2).

    A left record without coordinates, or with no right record nearby, contributes
    no pairs — that is FR-3's retention policy showing up here as an empty
    candidate set, not a special case this function has to know about.
    """
    index = _index_by_cell(right, cell_size)
    return [
        (record, candidate)
        for record in left
        for candidate in _candidates_for(record, index, cell_size)
    ]


PreparationPath = Literal["hoisted", "naive"]
"""Which call pattern produced a set of scores over candidate pairs
(ADR-0009, issue #65). ``"hoisted"`` is ``prepare_all`` ahead of the
comparison loop — the production pattern :func:`resolve` already uses
internally. ``"naive"`` is ``prepare`` called fresh inside the loop, once
per comparison, reproducing the redundant recomputation the hoist removes:
a record compared against several candidates is prepared that many times
over, never reused. Both are real, selectable code paths
(:func:`score_candidates`) rather than one path with a flag bolted on —
nothing defaults silently to either (see that function's required
``preparation`` argument)."""

_PREPARATION_PATHS: frozenset[str] = frozenset({"hoisted", "naive"})


@dataclass(frozen=True, slots=True)
class ScoredComparisons:
    """Every candidate pair's score, tagged with the preparation path that
    produced it (issue #65's third bullet: "the run record states which
    one produced each figure").

    The path travels on the value itself, the same way
    :class:`~joinless.measurement.WarmLatency` carries its own
    ``warmup_count``/``repetition_count`` rather than leaving a reader to
    infer how a figure was measured from context — a caller holding one
    ``ScoredComparisons`` never has to trust a variable name or a separate
    part of a record to know which call pattern produced its scores.

    That is also this issue's answer to where path attribution belongs in
    a run record's shape: **on the figure, not as a run-level flag.**
    Issue #66 measures preparation cost *both* ways, for the same arm, in
    the same run — a single "which path did this run use" field could not
    describe that run at all, since neither path is uniquely the run's
    path. Each figure is its own path's figure; this type is where that
    decision is made structural rather than left to be discovered when
    issue #66's own cost fields are added.
    """

    path: PreparationPath
    scores: tuple[float, ...]


def score_candidates(
    left: Sequence[Record],
    right: Sequence[Record],
    matcher: ThresholdMatcher[Any],
    *,
    preparation: PreparationPath,
    cell_size: float = DEFAULT_CELL_SIZE_DEGREES,
) -> ScoredComparisons:
    """Score every candidate pair (FR-2) under an explicitly chosen
    preparation strategy (ADR-0009, issue #65) — the naive per-comparison
    control, retained and selectable, next to the hoisted path
    :func:`resolve` always uses.

    ``preparation`` is a required keyword argument with no default:
    "neither path is the default by accident" (issue #65's third bullet)
    is enforced structurally here rather than left as a convention a
    caller could forget — there is no call to this function that omits
    which strategy it wants.

    ``"hoisted"`` calls ``matcher.scorer.prepare_all`` once for ``left``
    and once for ``right``, ahead of the comparison loop — :func:`resolve`'s
    own production pattern, reused here rather than reimplemented.
    ``"naive"`` calls ``matcher.scorer.prepare`` fresh for both sides of
    every candidate pair, reproducing the per-comparison recomputation
    ADR-0009 describes as "the naive implementation... most people write
    first."

    This function does not decide matches, break ties, or merge — unlike
    :func:`resolve`, whose job is exactly that. It exists only to make the
    two preparation call patterns comparable over the same candidate set:
    for the score-equality test issue #65's second bullet asks for, and as
    the primitive issue #66 times to measure preparation cost both ways.
    Returned scores are in :func:`candidate_pairs`'s own order.
    """
    if preparation not in _PREPARATION_PATHS:
        available = ", ".join(sorted(_PREPARATION_PATHS))
        raise ValueError(
            f"Unknown preparation path {preparation!r}. Available: {available}."
        )

    pairs = candidate_pairs(left, right, cell_size)

    if preparation == "hoisted":
        prepared_left: dict[int, Any] = dict(
            zip(
                (id(record) for record in left),
                matcher.scorer.prepare_all([record.name for record in left]),
                strict=True,
            )
        )
        prepared_right: dict[int, Any] = dict(
            zip(
                (id(record) for record in right),
                matcher.scorer.prepare_all([record.name for record in right]),
                strict=True,
            )
        )
        scores = tuple(
            matcher.scorer.score(
                prepared_left[id(left_record)], prepared_right[id(right_record)]
            )
            for left_record, right_record in pairs
        )
    else:
        scores = tuple(
            matcher.scorer.score(
                matcher.scorer.prepare(left_record.name),
                matcher.scorer.prepare(right_record.name),
            )
            for left_record, right_record in pairs
        )

    return ScoredComparisons(path=preparation, scores=scores)


def _haversine_km(a: Record, b: Record) -> float:
    """Great-circle distance between two records, in kilometres.

    Both records are expected to carry coordinates — every call site here reaches
    this function only after candidate generation has already restricted both sides
    to coordinate-bearing records — so a missing coordinate is treated as a caller
    error and raises, rather than silently standing in for a default distance that
    would make an unrelated pair look like the closest candidate.
    """
    if (
        a.latitude is None
        or a.longitude is None
        or b.latitude is None
        or b.longitude is None
    ):
        raise ValueError("_haversine_km requires coordinates on both records")

    lat1, lon1, lat2, lon2 = (
        math.radians(a.latitude),
        math.radians(a.longitude),
        math.radians(b.latitude),
        math.radians(b.longitude),
    )
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    haversine = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
    )
    return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(haversine))


def _populatedness(record: Record) -> int:
    """How much information a record carries, for the merge policy's default
    winner (FR-5): one point per non-empty ``fields`` value, one for a non-empty
    name, one for carrying coordinates at all."""
    score = sum(1 for value in record.fields.values() if value)
    if record.name:
        score += 1
    if record.latitude is not None and record.longitude is not None:
        score += 1
    return score


def _winner(a: Record, b: Record) -> Record:
    """The more-populated of two records (FR-5's default rule), with a
    content-derived tie-break so the choice never depends on which record a
    caller happened to pass first — ``_winner(a, b)`` and ``_winner(b, a)``
    always return the same one of the two."""
    score_a, score_b = _populatedness(a), _populatedness(b)
    if score_a != score_b:
        return a if score_a > score_b else b
    return a if record_id(a) < record_id(b) else b


def merge(a: Record, b: Record) -> MergedRecord:
    """Merge two matched records under the FR-5 policy.

    Three rules, in order of precedence:

    1. **Provenance is unioned.** ``sources`` names every source that contributed,
       always both of them — this one never has an exception.
    2. **Coordinates always win**, independent of rule 3: whichever record actually
       carries coordinates supplies them, even when it is the less-populated side.
       A record missing coordinates never overwrites a real pair with ``None``.
    3. **The more-populated record wins everything else** — its name, and its value
       for any ``fields`` key the two records disagree on. Keys present on only one
       side are kept regardless of which side is more populated; only a genuine
       disagreement invokes the rule.

    ``merge(a, b) == merge(b, a)``: every rule above is defined over the pair's
    content, never over argument order, so which record is passed first cannot
    change the result.
    """
    winner = _winner(a, b)
    loser = b if winner is a else a

    fields: dict[str, str] = dict(loser.fields)
    fields.update(winner.fields)

    if _has_coordinates(a) and _has_coordinates(b):
        latitude, longitude = winner.latitude, winner.longitude
    elif _has_coordinates(a):
        latitude, longitude = a.latitude, a.longitude
    elif _has_coordinates(b):
        latitude, longitude = b.latitude, b.longitude
    else:
        latitude, longitude = None, None

    return MergedRecord(
        name=winner.name,
        latitude=latitude,
        longitude=longitude,
        fields=MappingProxyType(fields),
        sources=frozenset({a.source, b.source}),
    )


def _select_best(
    left_record: Record,
    prepared_left: Any,
    candidates: Sequence[Record],
    prepared_by_id: Mapping[int, Any],
    matcher: ThresholdMatcher[Any],
) -> Record | None:
    """The candidate ``left_record`` matches best, or ``None`` if none clear the
    matcher's threshold.

    Ranked by score descending, then by distance ascending as the tie-break RFC-0001
    reserves for the resolver, then by the candidate's ``record_id`` so that a score
    *and* distance tie — two candidates at the same point scoring identically — still
    resolves to the same choice on every run rather than to iteration order.
    """
    best_record: Record | None = None
    best_key: tuple[float, float, str] | None = None
    for candidate in candidates:
        prepared_candidate = prepared_by_id[id(candidate)]
        if not matcher.matches(prepared_left, prepared_candidate):
            continue
        score = matcher.scorer.score(prepared_left, prepared_candidate)
        distance = _haversine_km(left_record, candidate)
        key = (-score, distance, record_id(candidate))
        if best_key is None or key < best_key:
            best_key = key
            best_record = candidate
    return best_record


def resolve(
    left: Sequence[Record],
    right: Sequence[Record],
    matcher: ThresholdMatcher[Any],
    *,
    cell_size: float = DEFAULT_CELL_SIZE_DEGREES,
) -> ResolutionResult:
    """Resolve two record sets with no shared key into one merged set (FR-1).

    The composition is identity (:mod:`joinless.records`) plus candidate generation
    (this module) plus scoring through the ``matcher`` seam (RFC-0001) plus the
    distance tie-break plus the merge policy — see the module docstring for how
    each stage is bounded from affecting the others. Preparation is batched
    (``prepare_all``) once per side, matching the production call pattern
    ADR-0009 requires; the naive per-comparison path is not used here.

    Swapping ``matcher.scorer`` changes only which candidates clear the threshold,
    and therefore which pairs form — candidate generation, the distance tie-break
    and the merge policy for whichever pairs do form are all independent of it.
    """
    right_index = _index_by_cell(right, cell_size)
    occupancy: Mapping[tuple[int, int], int] = {
        cell: len(bucket) for cell, bucket in right_index.items()
    }

    prepared_left = matcher.scorer.prepare_all([record.name for record in left])
    prepared_right = matcher.scorer.prepare_all([record.name for record in right])
    prepared_by_id: dict[int, Any] = {
        id(record): value for record, value in zip(right, prepared_right, strict=True)
    }

    pairs: list[MatchedPair] = []
    unmatched: list[UnmatchedRecord] = []
    matched_right_ids: set[int] = set()

    for record, prepared in zip(left, prepared_left, strict=True):
        if not _has_coordinates(record):
            unmatched.append(
                UnmatchedRecord(record=record, reason=_REASON_NO_COORDINATES)
            )
            continue

        candidates = _candidates_for(record, right_index, cell_size)
        if not candidates:
            unmatched.append(
                UnmatchedRecord(record=record, reason=_REASON_NO_CANDIDATES)
            )
            continue

        best = _select_best(record, prepared, candidates, prepared_by_id, matcher)
        if best is None:
            unmatched.append(
                UnmatchedRecord(record=record, reason=_REASON_NO_MATCH_ABOVE_THRESHOLD)
            )
            continue

        pairs.append(MatchedPair(left=record, right=best, merged=merge(record, best)))
        matched_right_ids.add(id(best))

    for record in right:
        if id(record) in matched_right_ids:
            continue
        if not _has_coordinates(record):
            unmatched.append(
                UnmatchedRecord(record=record, reason=_REASON_NO_COORDINATES)
            )
        else:
            unmatched.append(
                UnmatchedRecord(record=record, reason=_REASON_NOT_SELECTED)
            )

    return ResolutionResult(
        pairs=tuple(pairs), unmatched=tuple(unmatched), occupancy=occupancy
    )
