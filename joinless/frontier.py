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

**Why exactly three constraint dimensions.** RFC-0002 "Decision output" names them
directly: "a memory ceiling, a latency ceiling, an accuracy floor, any combination
of these." Peak resident memory and warm p50 latency are the two cost figures
:class:`~joinless.runrecord.ArmResult` measures once per arm, independent of
family; F1 is the one comparable accuracy figure RFC-0002's Metrics table names
("a single comparable number") and is read per family. A field left ``None``
means the reader stated no constraint on that dimension — it never excludes an
arm, and it is never coerced into an implicit ceiling of "whatever this run
happened to measure."

**Why undefined accuracy excludes an arm rather than passing or failing a
floor.** A family's F1 can be undefined — ``semantic alias`` and
``near-miss negative`` are all-negative by design (:mod:`joinless.corpus`'s module
docstring), so recall and F1 are undefined for every arm there, always. An
undefined F1 is not zero and is not "no constraint stated" either; comparing an
absent number against a floor is not a comparison this module can draw one way or
the other, so the arm is excluded from that family's frontier with the record's
own reason attached, the same "undefined propagates, never collapses to a number"
rule ADR-0013 states for every other figure this project reports.

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
    "Decision output"): a memory ceiling, a latency ceiling, an accuracy floor,
    any combination of these. Every field defaults to ``None``, meaning the
    reader stated no constraint on that dimension — never an implicit default
    that silently narrows the field to whatever this run happened to measure.

    Recorded verbatim on :class:`FrontierResult` (issue #70's fourth bullet:
    "constraints used are recorded with the output, so a frontier can be
    reproduced") — a caller holding a ``FrontierResult`` never has to trust
    that the constraints it was computed under match the ones asked for.
    """

    max_peak_rss_bytes: float | None = None
    max_warm_p50_seconds: float | None = None
    min_f1: float | None = None


@dataclass(frozen=True, slots=True)
class FrontierPoint:
    """One arm's operating point for one family: its own F1 at the run's
    frozen threshold (issue #96's "per-family operating point"), and its
    per-run cost figures alongside it — the exact three dimensions
    :class:`Constraints` can bound.
    """

    arm: str
    f1: float
    peak_rss_bytes: float
    warm_p50_seconds: float


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
    if f1 is None:
        return (
            None,
            f"F1 is undefined for {family!r}: {family_row['f1']['undefined_reason']}",
        )

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

    if constraints.min_f1 is not None and f1 < constraints.min_f1:
        return None, f"F1 {f1:.3f} is below the stated floor {constraints.min_f1:.3f}"
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

    return (
        FrontierPoint(
            arm=arm,
            f1=f1,
            peak_rss_bytes=peak_rss_bytes,
            warm_p50_seconds=warm_p50_seconds,
        ),
        None,
    )


def _dominates(candidate: FrontierPoint, other: FrontierPoint) -> bool:
    """Whether ``candidate`` dominates ``other``: at least as good on every
    dimension — higher or equal F1, lower or equal peak RSS, lower or equal
    warm p50 — and strictly better on at least one (RFC-0002 "Decision
    output": "no other arm beats it on every stated dimension at once").
    Two points equal on all three dimensions dominate neither one another,
    so both remain on the frontier — a genuine tie is not a winner either.
    """
    at_least_as_good = (
        candidate.f1 >= other.f1
        and candidate.peak_rss_bytes <= other.peak_rss_bytes
        and candidate.warm_p50_seconds <= other.warm_p50_seconds
    )
    strictly_better = (
        candidate.f1 > other.f1
        or candidate.peak_rss_bytes < other.peak_rss_bytes
        or candidate.warm_p50_seconds < other.warm_p50_seconds
    )
    return at_least_as_good and strictly_better


def _pareto_frontier(
    candidates: Sequence[FrontierPoint],
) -> tuple[tuple[FrontierPoint, ...], dict[str, str]]:
    """The non-dominated subset of ``candidates``, and the reason every
    dominated candidate was excluded — naming which surviving arm dominates
    it, so "excluded" never means only "failed a stated constraint."
    """
    frontier: list[FrontierPoint] = []
    dominated: dict[str, str] = {}
    for point in candidates:
        dominator = next(
            (
                other
                for other in candidates
                if other.arm != point.arm and _dominates(other, point)
            ),
            None,
        )
        if dominator is None:
            frontier.append(point)
        else:
            dominated[point.arm] = (
                f"dominated by {dominator.arm!r} (f1={dominator.f1:.3f}, "
                f"peak RSS={dominator.peak_rss_bytes:.0f} bytes, "
                f"warm p50={dominator.warm_p50_seconds:.9f}s)"
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
