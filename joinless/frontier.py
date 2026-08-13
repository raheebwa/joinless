# SPDX-License-Identifier: MIT
"""The accuracy-cost frontier under stated constraints (issue #70, RFC-0002
"Decision output", PRD MR-9, ADR-0010, issue #96).

**Why a frontier, never a winner.** The four arms are measured on dimensions that
share no unit — accuracy, warm latency, resident memory. Collapsing those into one
ranked row means choosing an exchange rate between an F1 point and a megabyte of
resident memory, and any rate this project picked would encode this project's
priorities, not the reader's (RFC-0002 "Decision output"). This module never
produces that row: it reports, per family and under whichever ceilings and floor
the caller states, which arms are not dominated — and "no arm qualifies" is a
first-class result, not an empty table (ADR-0013's fail-closed rule, applied
here: an empty tuple standing in for "nothing qualified" would look identical to
"nothing was checked").

**Why a record, not a scorer or a corpus.** Like :mod:`joinless.report`, this
module takes the plain, JSON-shaped record :func:`joinless.runrecord.record_to_dict`
produces — never the ``RunRecord`` dataclasses, never a scorer, never a corpus.
Nothing here re-measures or re-derives a figure the record does not already carry;
every number a frontier reports traces to the same run a reader could read straight
off the record themselves (RFC-0002: "``report`` is a pure function of the run
record" — the same discipline applies here, one level up).

**Why per family, never a whole-record frontier.** A reader choosing a matcher for
data resembling one family — ``character noise``, say — is not served by a
frontier computed from the aggregate, which pools every family's mixture together
and answers a question about *this benchmark's* composition rather than the
reader's own data (issue #96's decision comment). So the frontier is computed once
per family, from that family's own precision/recall/F1 at the run's frozen
threshold (:mod:`joinless.evaluation`'s ``accuracy.pooled.per_family``) — the
identical scoring procedure and identical threshold every arm already shares
(ADR-0011 rule 2), never a threshold re-fit per family. Reporting each arm's own
per-family figure this way is issue #96's "per-family operating points": not a
second round of per-arm tuning, but the same frozen-threshold result, broken out
by family instead of pooled, and set beside each arm's per-run cost figures so a
reader whose data resembles a family sees what that arm actually does there — a
regime the run's single aggregate index cannot show, but one this module can
compute purely by grouping data the record already holds.

**Why four constraint dimensions, not three (issue #106).** RFC-0002 "Decision
output" names a memory ceiling, a latency ceiling and an accuracy floor. Peak
resident memory and warm p50 latency are the two cost figures
:class:`~joinless.runrecord.ArmResult` measures once per arm, independent of
family; F1 is the one comparable accuracy figure RFC-0002's Metrics table names
("a single comparable number") and is read per family. Alongside those three sits
a fourth: false positives, read per family exactly as F1 is
(:mod:`joinless.evaluation`'s ``FamilyResult.false_positives`` — "a plain count,
never a ratio, so it has no empty-denominator case to be undefined by and is
defined on every family, every arm, every run"). Where F1 asks *how well did this
arm do*, false positives asks *how often did it call two different entities the
same* — a question issue #105 added because F1 cannot answer it on an
all-negative family, where precision and recall have no denominator at all. A
field left ``None`` means the reader stated no constraint on that dimension — it
never excludes an arm, and it is never coerced into an implicit ceiling of
"whatever this run happened to measure." Lower is better for false positives,
the same direction as the two cost ceilings; higher is better for F1 alone.

**Why undefined F1 still excludes an arm from a stated floor, but no longer
excludes it from the frontier outright (issue #106).** A family's F1 can be
undefined — ``semantic alias`` and ``near-miss negative`` are all-negative by
design (:mod:`joinless.corpus`'s module docstring), so recall and F1 are
undefined for every arm there, always; ``character noise`` can reach the same
state at run time, if every arm's calibrated threshold happens to admit nothing
on it. An undefined F1 is not zero and is not "no constraint stated" either;
comparing an absent number against a stated floor is not a comparison this
module can draw one way or the other, so a *stated* floor still excludes the arm,
with the record's own reason attached — the "undefined propagates, never
collapses to a number" rule ADR-0013 states for every other figure this project
reports. What changed is what happens when no floor is stated: false positives
is always defined (the paragraph above), so an arm with undefined F1 is no
longer dropped from the frontier for want of an axis to place it on — it competes
on false positives and cost like any other candidate, with ``f1=None`` recorded
on its :class:`FrontierPoint` rather than a number that was never measured.
Before this, a family where F1 was undefined for every arm reported "no arm
qualifies" under no constraints at all — indistinguishable from a constraint set
nothing satisfies, on the exact family where the run record's four arms produced
the most distinct outcomes (issue #106).

**What "dominates" means once F1 may be absent.** Domination stays what RFC-0002
defines it as — at least as good on every axis, strictly better on one — but an
axis that is undefined for a candidate cannot be compared, and ADR-0013's rule
against collapsing an undefined value into a number applies to a *comparison*
result exactly as it applies to a metric. Rather than decide, pair by pair,
whether to skip F1 (which would make "A beats B" and "B beats C" imply nothing
about "A beats C", breaking domination as an ordering and, with it, the
guarantee that a non-empty candidate set always yields a non-empty frontier),
this module decides once per family: F1 participates in every comparison in
that family's frontier if and only if every candidate reaching the domination
step has a defined F1, never a per-pair decision (:func:`_pareto_frontier`).
Every family in the current benchmark corpus is already uniform this way — F1's
undefinedness there traces to a family-wide property (no actual positives, or
every arm's threshold admitting nothing), never to one arm alone — so this rule
changes no comparison a defined-F1 family already made; it only decides the
family where F1 is undefined throughout, which is exactly the case issue #106
exists to fix.

**Why an unavailable or invalid arm is excluded, with its reason kept.**
Mirroring :mod:`joinless.report`: an arm whose accuracy is ``invalid``, or whose
accuracy, warm latency or peak memory is ``unavailable``, cannot be measured
against any constraint at all. It is excluded from the frontier, never silently
dropped from the record's own accounting — its reason travels into this family's
``excluded`` mapping exactly as it would into a rendered table row (ADR-0013).

**Why a dominated arm is excluded too, and why "no arm qualifies" is still
distinct from every arm being excluded for a constraint reason.** An arm that
passes every stated constraint but is beaten on every dimension at once by
another qualifying arm adds nothing a reader could not get from the arm that
beats it — it is excluded from the frontier with a reason naming which arm
dominates it. Whether a family's frontier is empty because every arm failed a
stated constraint, because every comparable arm's accuracy was undefined, or
because none of that family's arms exist in this run at all, the result is the
same explicit :class:`NoArmQualifies` value, carrying every arm's own reason —
never an empty tuple a caller could mistake for "nothing was checked."
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True)
class Constraints:
    """The reader-stated constraints a frontier is computed under (RFC-0002
    "Decision output", extended by issue #106): a memory ceiling, a latency
    ceiling, a false-positives ceiling, an accuracy floor, any combination of
    these. Every field defaults to ``None``, meaning the reader stated no
    constraint on that dimension — never an implicit default that silently
    narrows the field to whatever this run happened to measure.

    Recorded verbatim on :class:`FrontierResult` (issue #70's fourth bullet:
    "constraints used are recorded with the output, so a frontier can be
    reproduced") — a caller holding a ``FrontierResult`` never has to trust
    that the constraints it was computed under match the ones asked for.
    """

    max_peak_rss_bytes: float | None = None
    max_warm_p50_seconds: float | None = None
    min_f1: float | None = None
    max_false_positives: int | None = None


@dataclass(frozen=True, slots=True)
class FrontierPoint:
    """One arm's operating point for one family: its own F1 at the run's
    frozen threshold (issue #96's "per-family operating point"), its false
    positives for that same family (issue #106), and its per-run cost figures
    alongside them — the exact four dimensions :class:`Constraints` can bound.

    ``f1`` is ``None`` exactly when this family's F1 is undefined for this arm
    — never a number standing in for "not measured" (module docstring).
    ``false_positives`` is never ``None``: it is a plain count with no
    empty-denominator case to be undefined by, defined on every family, every
    arm, every run (:mod:`joinless.evaluation`'s ``FamilyResult.false_positives``).
    """

    arm: str
    f1: float | None
    peak_rss_bytes: float
    warm_p50_seconds: float
    false_positives: int


@dataclass(frozen=True, slots=True)
class NoArmQualifies:
    """The explicit "no arm qualifies" result (issue #70's second bullet): a
    constraint set nothing in this family satisfies. ``reason`` names every
    arm this family considered and why each one did not make the frontier —
    the same text that would sit in the family's own ``excluded`` mapping,
    joined into one sentence so a reader of just this value, with no other
    context, still sees why.
    """

    reason: str


@dataclass(frozen=True, slots=True)
class FamilyFrontier:
    """One family's frontier: either the arms that qualify and are not
    dominated (:class:`FrontierPoint`, ...), or :class:`NoArmQualifies`.

    ``excluded`` names every arm in this family that is *not* in ``frontier``
    — whether it failed a stated constraint, reported an undefined or
    unavailable figure, or was dominated by an arm that did qualify — mapped
    to the one reason it is missing (mirrors how :mod:`joinless.report` keeps
    an unavailable arm's row rather than dropping it). When ``frontier`` is
    :class:`NoArmQualifies`, ``excluded`` names every arm in the family;
    otherwise it names every arm not on the frontier.
    """

    family: str
    frontier: tuple[FrontierPoint, ...] | NoArmQualifies
    excluded: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class FrontierResult:
    """The frontier for every family in a record, under one set of
    constraints (issue #70) — never a whole-record frontier, and never a
    single row naming a winner anywhere in it (module docstring).
    """

    constraints: Constraints
    per_family: tuple[FamilyFrontier, ...]


def _family_order(results: Mapping[str, Any]) -> list[str]:
    """The family order this record's "ok" arms share — the first "ok" arm's
    own pooled per-family family list.

    Mirrors :func:`joinless.report._family_order` line for line rather than
    importing it: both modules read the same plain JSON-shaped record but
    otherwise depend on nothing else in this project (each module's own
    docstring), and importing one into the other would couple two readers
    that issue #70 and issue #71 each built to stand alone.
    """
    for arm_result in results.values():
        accuracy = arm_result["accuracy"]
        if accuracy.get("status") == "ok":
            return [row["family"] for row in accuracy["pooled"]["per_family"]]
    return []


def _evaluate_candidate(
    arm: str,
    arm_result: Mapping[str, Any],
    family: str,
    constraints: Constraints,
) -> tuple[FrontierPoint | None, str | None]:
    """``arm``'s operating point for ``family`` under ``constraints``, or the
    one reason it does not qualify — never both, mirroring the same
    value-or-reason invariant :class:`~joinless.evaluation.Metric` enforces
    for a single figure, applied here to a whole candidate.
    """
    accuracy = arm_result["accuracy"]
    if accuracy.get("status") != "ok":
        return None, f"accuracy is {accuracy['status']} — {accuracy['reason']}"

    family_row = next(
        (row for row in accuracy["pooled"]["per_family"] if row["family"] == family),
        None,
    )
    if family_row is None:
        return None, f"{family!r} was not reported in this arm's accuracy"

    f1 = family_row["f1"]["value"]
    false_positives = family_row["false_positives"]

    peak_memory = arm_result["peak_memory"]
    if peak_memory.get("status") != "ok":
        return None, f"peak memory is {peak_memory['status']} — {peak_memory['reason']}"
    peak_rss_bytes = peak_memory["peak_rss_bytes"]

    warm_latency = arm_result["warm_latency"]
    if warm_latency.get("status") != "ok":
        return (
            None,
            f"warm latency is {warm_latency['status']} — {warm_latency['reason']}",
        )
    warm_p50_seconds = warm_latency["p50_seconds"]

    # A *stated* accuracy floor still excludes an arm whose F1 is undefined —
    # an absent number cannot be compared against a floor one way or the
    # other (module docstring, ADR-0013). With no floor stated, undefined F1
    # is not itself a reason to exclude: the arm still competes on false
    # positives and cost (issue #106).
    if constraints.min_f1 is not None:
        if f1 is None:
            return None, (
                f"F1 is undefined for {family!r}: "
                f"{family_row['f1']['undefined_reason']} — cannot be compared "
                "against the stated accuracy floor"
            )
        if f1 < constraints.min_f1:
            return (
                None,
                f"F1 {f1:.3f} is below the stated floor {constraints.min_f1:.3f}",
            )
    if (
        constraints.max_peak_rss_bytes is not None
        and peak_rss_bytes > constraints.max_peak_rss_bytes
    ):
        return None, (
            f"peak RSS {peak_rss_bytes:.0f} bytes exceeds the stated ceiling "
            f"{constraints.max_peak_rss_bytes:.0f} bytes"
        )
    if (
        constraints.max_warm_p50_seconds is not None
        and warm_p50_seconds > constraints.max_warm_p50_seconds
    ):
        return None, (
            f"warm p50 {warm_p50_seconds:.9f}s exceeds the stated ceiling "
            f"{constraints.max_warm_p50_seconds:.9f}s"
        )
    if (
        constraints.max_false_positives is not None
        and false_positives > constraints.max_false_positives
    ):
        return None, (
            f"false positives {false_positives} exceeds the stated ceiling "
            f"{constraints.max_false_positives}"
        )

    return (
        FrontierPoint(
            arm=arm,
            f1=f1,
            peak_rss_bytes=peak_rss_bytes,
            warm_p50_seconds=warm_p50_seconds,
            false_positives=false_positives,
        ),
        None,
    )


def _dominates(
    candidate: FrontierPoint, other: FrontierPoint, *, compare_f1: bool = True
) -> bool:
    """Whether ``candidate`` dominates ``other``: at least as good on every
    dimension — higher or equal F1, lower or equal peak RSS, lower or equal
    warm p50, lower or equal false positives — and strictly better on at
    least one (RFC-0002 "Decision output": "no other arm beats it on every
    stated dimension at once", extended by issue #106 to a fourth axis). Two
    points equal on every dimension dominate neither one another, so both
    remain on the frontier — a genuine tie is not a winner either.

    ``compare_f1`` states whether F1 is one of the axes being compared at
    all. It defaults to ``True`` — the ordinary case, where every candidate
    in the comparison has a defined F1. A caller passes ``False`` only when
    F1 is undefined for the candidates being compared, in which case it is
    excluded from the comparison entirely rather than treated as tying or
    losing (module docstring's "what dominates means once F1 may be
    absent"): peak RSS, warm p50 and false positives alone decide it.
    """
    at_least_as_good = (
        candidate.peak_rss_bytes <= other.peak_rss_bytes
        and candidate.warm_p50_seconds <= other.warm_p50_seconds
        and candidate.false_positives <= other.false_positives
    )
    strictly_better = (
        candidate.peak_rss_bytes < other.peak_rss_bytes
        or candidate.warm_p50_seconds < other.warm_p50_seconds
        or candidate.false_positives < other.false_positives
    )
    if compare_f1:
        assert candidate.f1 is not None and other.f1 is not None, (
            "compare_f1=True requires both points to carry a defined F1 — "
            "_pareto_frontier only sets it once every candidate in the "
            "family does"
        )
        at_least_as_good = at_least_as_good and candidate.f1 >= other.f1
        strictly_better = strictly_better or candidate.f1 > other.f1
    return at_least_as_good and strictly_better


def format_mb(bytes_value: float) -> str:
    """Bytes as megabytes, one decimal — the rendering a reader meets.

    Lives here rather than in a presentation module because :func:`_describe`
    needs it: a frontier's own exclusion reasons are published verbatim beside
    tables that use this same rendering, and two spellings of one quantity in
    adjacent lines make a reader convert between them to check they agree.
    """
    return f"{bytes_value / 1_000_000:.1f} MB"


def format_microseconds(seconds: float) -> str:
    """Seconds as microseconds, two decimals — see :func:`format_mb`."""
    return f"{seconds * 1_000_000:.2f}µs"


def _describe(point: FrontierPoint, *, compare_f1: bool) -> str:
    f1_part = f"f1={point.f1:.3f}, " if compare_f1 else ""
    return (
        f"{f1_part}false_positives={point.false_positives}, "
        f"peak RSS={format_mb(point.peak_rss_bytes)}, "
        f"warm p50={format_microseconds(point.warm_p50_seconds)}"
    )


def _pareto_frontier(
    candidates: Sequence[FrontierPoint],
) -> tuple[tuple[FrontierPoint, ...], dict[str, str]]:
    """The non-dominated subset of ``candidates``, and the reason every
    dominated candidate was excluded — naming which surviving arm dominates
    it, so "excluded" never means only "failed a stated constraint."

    F1 participates in domination for this whole family if and only if every
    candidate here has a defined F1 — decided once, not per pair (module
    docstring's "what dominates means once F1 may be absent"; the same
    decision is what keeps domination a strict partial order, which is what
    guarantees a non-empty ``candidates`` always yields a non-empty
    frontier — see the comment where that guarantee is used, below).
    """
    compare_f1 = all(point.f1 is not None for point in candidates)
    frontier: list[FrontierPoint] = []
    dominated: dict[str, str] = {}
    for point in candidates:
        dominator = next(
            (
                other
                for other in candidates
                if other.arm != point.arm
                and _dominates(other, point, compare_f1=compare_f1)
            ),
            None,
        )
        if dominator is None:
            frontier.append(point)
        else:
            dominated[point.arm] = (
                f"dominated by {dominator.arm!r} "
                f"({_describe(dominator, compare_f1=compare_f1)})"
            )
    return tuple(frontier), dominated


def _family_frontier(
    family: str, results: Mapping[str, Any], constraints: Constraints
) -> FamilyFrontier:
    candidates: list[FrontierPoint] = []
    excluded: dict[str, str] = {}
    for arm, arm_result in results.items():
        point, reason = _evaluate_candidate(arm, arm_result, family, constraints)
        if point is not None:
            candidates.append(point)
        else:
            assert reason is not None
            excluded[arm] = reason

    frontier_points, dominated = _pareto_frontier(candidates)
    excluded.update(dominated)

    if not frontier_points:
        # A non-empty `candidates` always yields a non-empty Pareto frontier
        # (domination is a strict partial order over a finite set, so a
        # maximal element always exists) — so reaching here with `results`
        # non-empty means `excluded` already names every arm in it.
        reason = "; ".join(f"{arm} ({why})" for arm, why in excluded.items())
        return FamilyFrontier(
            family=family,
            frontier=NoArmQualifies(reason=reason),
            excluded=MappingProxyType(excluded),
        )
    return FamilyFrontier(
        family=family,
        frontier=frontier_points,
        excluded=MappingProxyType(excluded),
    )


def compute_frontier(
    record: Mapping[str, Any], constraints: Constraints
) -> FrontierResult:
    """The accuracy-cost frontier for every family in ``record``, under
    ``constraints`` (issue #70).

    ``record`` is the plain JSON-shaped record :func:`joinless.runrecord.record_to_dict`
    produces and :func:`json.load` reads back — this function re-measures
    nothing and accepts no figure through any argument beyond the two here
    (module docstring). Family order follows the record's own accuracy
    reports (:func:`_family_order`); a record with no "ok" arm at all yields
    an empty ``per_family`` tuple, since there is no family to compute a
    frontier for.
    """
    results = record["results"]
    per_family = tuple(
        _family_frontier(family, results, constraints)
        for family in _family_order(results)
    )
    return FrontierResult(constraints=constraints, per_family=per_family)
