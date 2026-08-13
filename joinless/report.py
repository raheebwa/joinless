# SPDX-License-Identifier: MIT
"""The per-family results table (issue #71, RFC-0002 "Output", PRD MR-2 and
success criterion 4).

**Renders a plain, JSON-shaped record — not the ``RunRecord`` dataclasses.**
:func:`joinless.runrecord.record_to_dict` is the one place a ``RunRecord``
becomes JSON, and :func:`json.load` is the one place ``joinless report``
reads a record back — this module renders exactly that shape, the same one
``benchmarks/*.json`` already holds on disk. Building a second, parallel
deserialiser back into the dataclasses only to immediately destructure it
again for formatting would be a second representation of the same record,
with its own chance to drift from the one actually written (RFC-0002's own
"a report that can drift from the record it claims to summarise" is exactly
the failure this module's own boundary is drawn against). A plain
``Mapping`` is also what makes this module importable, and callable, without
pulling in a single line of :mod:`joinless.evaluation`, :mod:`joinless.scoring`
or :mod:`joinless.measurement` — nothing here constructs a scorer, times
anything, or reads a file; it only formats values a caller already has
(issue #46: "report... performs no measurement and initialises no arm").

**One block per perturbation family, arms down, metrics across** — the shape
issue #71 names directly: "the table a reader actually reads." Family order
comes from the first arm in ``results`` whose accuracy is ``"ok"`` (Python's
own JSON object insertion order, preserved by :func:`json.load`) — every
"ok" arm in one run reports the same families, drawn from the same pooled
corpus (:func:`joinless.evaluation.evaluate`), so there is exactly one order
to pick among them, not a choice this module makes per arm.

**An unavailable arm keeps its row, in every family block and in the
aggregate section, carrying its reason** (ADR-0013) — never dropped, and
never rendered as if it had produced zeros. **An undefined metric renders
`null (<reason>)`** — never `0` and never blank — the same ``Metric``
distinction :mod:`joinless.evaluation` already enforces at construction time,
carried through to the page a reader actually looks at.

**The aggregate is labelled as derived** (issue #71's second bullet) by
printing the record's own ``aggregate.derivation`` sentence once, as a
caption over the aggregate section, rather than recomputing anything: the
number and the sentence explaining where it came from are read off the same
field :func:`joinless.evaluation._aggregate` already wrote there.

**Contradicted families are marked** on the family's own heading line, from
the record's own ``contradictions`` list (:func:`joinless.evaluation.find_contradictions`'s
output, persisted once by :mod:`joinless.cli` — never recomputed here, for the
same "computed once, never twice" reason ``contradictions`` itself is computed
once and handed to :meth:`~joinless.runrecord.RunAssembly.build` rather than
re-derived at render time).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

# Wide enough that "null (no predicted positives)" — the longest undefined
# reason this module's own callers produce (ADR-0013's two ``_ratio`` reasons)
# — never collides with the next column. A fixed width, not a per-render
# computed one: this is plain terminal text, not an artifact table, and a
# reader gains nothing from perfectly tight columns that a fixed width does
# not already give them.
_ARM_COLUMN_WIDTH = 14
_METRIC_COLUMN_WIDTH = 32


def _format_metric(metric: Mapping[str, Any]) -> str:
    """One ``Metric`` as printed text (ADR-0013): the value to three decimal
    places, or ``null`` with the reason it has none — never ``0`` standing in
    for "no denominator," and never blank standing in for "nothing to show."
    """
    value = metric["value"]
    if value is not None:
        return f"{value:.3f}"
    return f"null ({metric['undefined_reason']})"


def _row(label: str, cells: Sequence[str]) -> str:
    """One table row, columns joined with an explicit two-space gap rather
    than relying on ``ljust`` alone: an undefined metric's reason (``"null
    (precision is undefined: no predicted positives)"``) can exceed
    ``_METRIC_COLUMN_WIDTH`` on its own, and ``ljust`` adds no padding once a
    cell is already at or past its target width — a plain ``"".join`` would
    then glue that cell straight onto the next column's text with no
    separator at all. The two-space join guarantees a gap regardless of how
    long any one cell's undefined reason runs.
    """
    padded = [label.ljust(_ARM_COLUMN_WIDTH)]
    padded.extend(cell.ljust(_METRIC_COLUMN_WIDTH) for cell in cells)
    return "  " + "  ".join(padded)


def _unavailable_row(arm: str, accuracy: Mapping[str, Any]) -> str:
    return f"  {arm.ljust(_ARM_COLUMN_WIDTH)}unavailable — {accuracy['reason']}"


def _family_order(results: Mapping[str, Any]) -> list[str]:
    """The family order every "ok" arm in ``results`` already shares (module
    docstring) — the first one found, in ``results``'s own iteration order.
    An empty list means no arm in this record produced a comparable
    accuracy report at all; the per-family section is then simply empty and
    the aggregate section (which handles every non-"ok" arm on its own) is
    where every arm's row, and its reason, still appears.
    """
    for arm_result in results.values():
        accuracy = arm_result["accuracy"]
        if accuracy.get("status") == "ok":
            return [row["family"] for row in accuracy["pooled"]["per_family"]]
    return []


def _derivation(results: Mapping[str, Any]) -> str:
    for arm_result in results.values():
        accuracy = arm_result["accuracy"]
        if accuracy.get("status") == "ok":
            return str(accuracy["pooled"]["aggregate"]["derivation"])
    return "no arm in this record produced a comparable aggregate to derive"


def _contradiction_marker(
    family: str, contradictions: Sequence[Mapping[str, Any]]
) -> str:
    contradiction = next((c for c in contradictions if c["family"] == family), None)
    if contradiction is None:
        return ""
    actual = ", ".join(repr(arm) for arm in contradiction["actual_winners"])
    return (
        f"  [CONTRADICTED — expected {contradiction['expected_winner']!r} to win; "
        f"actual winner(s): {actual}]"
    )


def _family_block(
    family: str,
    results: Mapping[str, Any],
    contradictions: Sequence[Mapping[str, Any]],
) -> list[str]:
    lines = [f"family: {family}{_contradiction_marker(family, contradictions)}"]
    lines.append(_row("arm", ("precision", "recall", "f1", "false_positives")))
    for arm, arm_result in results.items():
        accuracy = arm_result["accuracy"]
        if accuracy.get("status") != "ok":
            lines.append(_unavailable_row(arm, accuracy))
            continue
        family_rows = {row["family"]: row for row in accuracy["pooled"]["per_family"]}
        row = family_rows[family]
        lines.append(
            _row(
                arm,
                (
                    _format_metric(row["precision"]),
                    _format_metric(row["recall"]),
                    _format_metric(row["f1"]),
                    str(row["false_positives"]),
                ),
            )
        )
    lines.append("")
    return lines


def _aggregate_section(results: Mapping[str, Any]) -> list[str]:
    lines = [f"aggregate — derived: {_derivation(results)}"]
    lines.append(_row("arm", ("precision", "recall", "f1")))
    for arm, arm_result in results.items():
        accuracy = arm_result["accuracy"]
        if accuracy.get("status") != "ok":
            lines.append(_unavailable_row(arm, accuracy))
            continue
        aggregate = accuracy["pooled"]["aggregate"]
        lines.append(
            _row(
                arm,
                (
                    _format_metric(aggregate["precision"]),
                    _format_metric(aggregate["recall"]),
                    _format_metric(aggregate["f1"]),
                ),
            )
        )
    return lines


class UnsupportedSchema(ValueError):
    """A record this build does not know how to render.

    The record is not malformed — it is a faithful record of the run that
    produced it, written by an earlier version of the schema. What it lacks is
    a block this renderer reads, so refusing is correct and rendering a partial
    table would be worse. Only the shape of the refusal matters here: a reader
    is told which schema their record carries and which one this build renders,
    rather than meeting an internal dictionary key in a stack trace.
    """


# The one schema this renderer understands. `joinless.cli` binds its `_SCHEMA` to
# this value rather than repeating it, so a bump cannot move the writer while
# leaving the reader behind — the failure that shape prevents is a record written
# and immediately unreadable by the same build.
RENDERABLE_SCHEMA = "benchmark-v6"


def render_per_family_table(record: Mapping[str, Any]) -> str:
    """The per-family results table for ``record`` (issue #71) — arms down,
    metrics across, one block per perturbation family, an aggregate section
    labelled as derived from that same table, unavailable arms kept in every
    block with their reason, and contradicted families marked on their own
    heading.

    ``record`` is the plain JSON-shaped record :func:`joinless.runrecord.record_to_dict`
    produces and :func:`json.load` reads back — module docstring. Nothing
    here re-measures or accepts a metric through any argument beyond this
    one already-written record (issue #46).
    """
    schema = record["schema"]
    if schema != RENDERABLE_SCHEMA:
        raise UnsupportedSchema(
            f"record carries schema {schema!r}; this build renders "
            f"{RENDERABLE_SCHEMA!r}. Re-run `joinless benchmark` to produce a "
            "record this version can read."
        )
    lines = [f"run {record['record_id']} (schema {record['schema']})", ""]
    results = record["results"]
    contradictions = record["contradictions"]
    for family in _family_order(results):
        lines.extend(_family_block(family, results, contradictions))
    lines.extend(_aggregate_section(results))
    return "\n".join(lines) + "\n"
