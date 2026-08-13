# SPDX-License-Identifier: MIT
"""The README's generated results section (issue #72, epic #69, ADR-0010,
RFC-0002 "Decision output").

**Why generated, not typed.** Epic #69's rule is that a number that cannot be
reproduced from a committed command does not appear in the README. A typed
figure can drift from the record that produced it the moment either one is
edited alone; a generated one cannot, because there is only one place either
figure lives. This module is that one place: it turns the plain, JSON-shaped
record :func:`joinless.runrecord.record_to_dict` produces and :func:`json.load`
reads back — the same shape :mod:`joinless.report` and :mod:`joinless.frontier`
already render/compute over — into the markdown section
``scripts/render_readme_results.py`` splices into ``README.md``. Like both of
those modules, this one performs no measurement and initialises no arm; every
figure it prints is read off the one record a caller hands it.

**Why the frontier, not a typed "reasonable choice."** RFC-0002 "Decision
output" rejects a single ranked winner for the same reason ADR-0010 rejects
"the classical/neural crossover": collapsing accuracy, cost and false
positives into one row means choosing an exchange rate this project has no
standing to fix on a reader's behalf. So "which arm is the reasonable choice"
per family (epic #69) is answered exactly as :mod:`joinless.frontier` already
answers it — the Pareto frontier under no stated constraints, computed by
:func:`joinless.frontier.compute_frontier` and printed verbatim, arm names and
exclusion reasons alike. This module adds no second notion of "reasonable";
it renders the one the project already computes and tests.

**Why the record cites itself.** :func:`joinless.runrecord.build_record_id`
states plainly that a record's ``record_id`` *is* the file name
:func:`~joinless.runrecord.write_record` writes it under — so
``benchmarks/<record_id>`` is the citation, derived from the record's own
field rather than threaded through as a second argument a caller could pass
out of sync with the record actually being rendered.

**Why one schema.** This module reuses :data:`joinless.report.RENDERABLE_SCHEMA`
and :class:`joinless.report.UnsupportedSchema` rather than declaring a second
schema string: a record this build cannot render as a terminal table is a
record it cannot render as a README section either, for the identical reason
— refusing is correct, and a second, independently-maintained schema constant
is exactly the kind of drift RFC-0002's "a report that can drift from the
record it claims to summarise" warns against, one level up.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from joinless.frontier import (
    Constraints,
    FrontierPoint,
    NoArmQualifies,
    compute_frontier,
    format_mb,
    format_microseconds,
)
from joinless.report import RENDERABLE_SCHEMA, UnsupportedSchema

# The scope statement below paraphrases ADR-0010's "Supported" and "Not
# claimed" tables rather than quoting the ADR file at render time: a run
# record carries no pointer back into docs/adrs/, and reading the ADR file
# from disk here would make this module's output depend on which commit of a
# *different* file happens to be checked out, rather than being a pure
# function of the one record it is handed (module docstring). The boundary
# stated is policy text, not a measured figure — ADR-0010 itself is the
# source of truth for it, and this string is checked against that ADR by
# review, not by re-reading the file at generation time.
_SCOPE_STATEMENT = (
    "Results are scoped to this disclosed synthetic benchmark on the "
    "reference machine named above. They describe how these four matchers "
    "compare on this corpus, what the embedding arms cost on this hardware, "
    "and whether the preparation hoist and quantization change those costs "
    "— nothing wider (ADR-0010). They are not a universal ranking. In "
    'particular: this is not a location of "the classical/neural '
    'crossover" — two classical matchers and one embedding model do not '
    "locate a frontier across either family of technique; it is not a claim "
    "that embeddings beat string matching in general, which is already "
    "established prior art (see LinkTransformer, cited below), not "
    "re-asserted here; and it is not a claim that these results transfer to "
    "real corpora unchanged, since the corpus is synthetic by construction."
)

_NO_CONSTRAINTS = Constraints()

# The marker pair `scripts/render_readme_results.py` splices generated
# content between. Public (module docstring names the script as this
# module's one caller) so the script imports the exact strings it writes
# rather than a second copy of them that could drift from what this module's
# own tests assert against.
MARKER_BEGIN = (
    "<!-- BEGIN GENERATED RESULTS: do not edit by hand — regenerate with "
    "`uv run python scripts/render_readme_results.py <record> README.md` -->"
)
MARKER_END = "<!-- END GENERATED RESULTS -->"


class MissingMarkers(ValueError):
    """``README.md`` carries no ``MARKER_BEGIN``/``MARKER_END`` pair for
    :func:`splice_into_readme` to replace the content between. Refusing here
    is the same fail-closed choice :class:`joinless.report.UnsupportedSchema`
    makes for a record this build cannot read: a README with no marker pair
    is a README this tool cannot update without guessing where the generated
    section belongs, and guessing is worse than refusing.
    """


def _format_metric(metric: Mapping[str, Any]) -> str:
    """One ``Metric`` (value/undefined_reason) as printed text (ADR-0013) —
    a small private formatter, deliberately not imported from
    :mod:`joinless.report`, mirroring that module's own choice to duplicate
    rather than import :mod:`joinless.frontier`'s ``_family_order``: each
    reader of the same plain record shape stands on its own (that module's
    docstring)."""
    value = metric["value"]
    if value is not None:
        return f"{value:.3f}"
    return f"null ({metric['undefined_reason']})"


def _format_warm_latency(warm_latency: Mapping[str, Any]) -> str:
    if warm_latency.get("status") != "ok":
        return f"unavailable — {warm_latency['reason']}"
    return format_microseconds(warm_latency["p50_seconds"])


def _format_peak_memory(peak_memory: Mapping[str, Any]) -> str:
    if peak_memory.get("status") != "ok":
        return f"unavailable — {peak_memory['reason']}"
    return format_mb(peak_memory["peak_rss_bytes"])


def _format_artifact_size(artifact_size: Mapping[str, Any]) -> str:
    value = artifact_size["value"]
    if value is None:
        return f"— ({artifact_size['undefined_reason']})"
    return format_mb(value)


def _describe_hardware(hardware: Mapping[str, Any]) -> str:
    ram_gib = hardware["total_memory_bytes"] / (1024**3)
    return (
        f"{hardware['system']} {hardware['release']} ({hardware['machine']}), "
        f"{hardware['cpu_count']} cores, {ram_gib:.1f} GiB RAM, "
        f"Python {hardware['python_version']}"
    )


def _describe_corpus(evaluation_set: Mapping[str, Any]) -> str:
    seeds = ", ".join(str(seed) for seed in evaluation_set["seeds"])
    families = ", ".join(sorted(evaluation_set["case_mixture"]))
    return (
        "the built-in synthetic corpus (`joinless.corpus`), pooled across "
        f"seeds {seeds}; families: {families}"
    )


def _aggregate_table(results: Mapping[str, Any]) -> list[str]:
    lines = [
        "| arm | aggregate F1 | warm p50 | peak RSS | artifact |",
        "|---|---|---|---|---|",
    ]
    for arm, arm_result in results.items():
        accuracy = arm_result["accuracy"]
        if accuracy.get("status") != "ok":
            lines.append(f"| `{arm}` | unavailable — {accuracy['reason']} | | | |")
            continue
        f1 = _format_metric(accuracy["pooled"]["aggregate"]["f1"])
        warm = _format_warm_latency(arm_result["warm_latency"])
        peak = _format_peak_memory(arm_result["peak_memory"])
        artifact = _format_artifact_size(arm_result["artifact_size"])
        lines.append(f"| `{arm}` | {f1} | {warm} | {peak} | {artifact} |")
    return lines


def _family_table(family: str, results: Mapping[str, Any]) -> list[str]:
    lines = [
        "| arm | f1 | false_positives | warm p50 | peak RSS |",
        "|---|---|---|---|---|",
    ]
    for arm, arm_result in results.items():
        accuracy = arm_result["accuracy"]
        if accuracy.get("status") != "ok":
            lines.append(f"| `{arm}` | unavailable — {accuracy['reason']} | | | |")
            continue
        row = next(r for r in accuracy["pooled"]["per_family"] if r["family"] == family)
        f1 = _format_metric(row["f1"])
        false_positives = str(row["false_positives"])
        warm = _format_warm_latency(arm_result["warm_latency"])
        peak = _format_peak_memory(arm_result["peak_memory"])
        lines.append(f"| `{arm}` | {f1} | {false_positives} | {warm} | {peak} |")
    return lines


def _describe_frontier_point(point: FrontierPoint) -> str:
    f1 = "null" if point.f1 is None else f"{point.f1:.3f}"
    return f"`{point.arm}` (f1={f1}, false_positives={point.false_positives})"


def _frontier_line(frontier: tuple[FrontierPoint, ...] | NoArmQualifies) -> str:
    if isinstance(frontier, NoArmQualifies):
        return f"**No arm qualifies** (no stated constraints): {frontier.reason}"
    points = ", ".join(_describe_frontier_point(point) for point in frontier)
    return f"**On the frontier** (no stated constraints — none of these is dominated by another): {points}."


def _excluded_lines(excluded: Mapping[str, str]) -> list[str]:
    return [f"- `{arm}`: {reason}" for arm, reason in excluded.items()]


def render_results_section(record: Mapping[str, Any]) -> str:
    """The README's generated results section for ``record`` (issue #72).

    Names the record it was generated from, the reference machine, the
    corpus and the seeds (epic #69's own bullets), states ADR-0010's claim
    boundary, reports the aggregate table, and — per family — the same table
    plus the Pareto frontier :mod:`joinless.frontier` already computes under
    no stated constraints (module docstring: "why the frontier, not a typed
    'reasonable choice'").
    """
    schema = record["schema"]
    if schema != RENDERABLE_SCHEMA:
        raise UnsupportedSchema(
            f"record carries schema {schema!r}; this build renders "
            f"{RENDERABLE_SCHEMA!r}. Re-run `joinless benchmark` to produce a "
            "record this version can read."
        )

    record_id = record["record_id"]
    results = record["results"]
    hardware = record["environment"]["hardware"]
    evaluation_set = record["evaluation_set"]

    lines = [
        "## Results",
        "",
        (
            f"Generated from [`benchmarks/{record_id}`](benchmarks/{record_id}) by "
            f"`uv run python scripts/render_readme_results.py benchmarks/{record_id} "
            "README.md` — every figure below traces to that one run record."
        ),
        "",
        f"**Reference machine:** {_describe_hardware(hardware)}",
        "",
        f"**Corpus:** {_describe_corpus(evaluation_set)}",
        "",
        _SCOPE_STATEMENT,
        "",
        "### Aggregate",
        "",
        *_aggregate_table(results),
        "",
    ]

    frontier_result = compute_frontier(record, _NO_CONSTRAINTS)
    for family_frontier in frontier_result.per_family:
        lines.append(f"### {family_frontier.family}")
        lines.append("")
        lines.extend(_family_table(family_frontier.family, results))
        lines.append("")
        lines.append(_frontier_line(family_frontier.frontier))
        excluded_lines = _excluded_lines(family_frontier.excluded)
        if excluded_lines:
            lines.append("")
            lines.extend(excluded_lines)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def splice_into_readme(readme_text: str, section: str) -> str:
    """``readme_text`` with everything between :data:`MARKER_BEGIN` and
    :data:`MARKER_END` replaced by ``section`` (issue #72's first bullet:
    "generated ... by a committed command", not typed in place by hand).

    Idempotent: splicing the same ``section`` into the result of a previous
    splice reproduces it exactly, since the replaced region is always
    bounded by the same two markers, never by whatever the previous
    generated content happened to contain. Raises :class:`MissingMarkers`
    rather than appending or guessing a location when the pair is absent.
    """
    if MARKER_BEGIN not in readme_text or MARKER_END not in readme_text:
        raise MissingMarkers(
            f"README.md is missing the marker pair this tool splices "
            f"generated content between: {MARKER_BEGIN!r} ... {MARKER_END!r}"
        )
    before, _, rest = readme_text.partition(MARKER_BEGIN)
    _, _, after = rest.partition(MARKER_END)
    return f"{before}{MARKER_BEGIN}\n\n{section}\n{MARKER_END}{after}"
