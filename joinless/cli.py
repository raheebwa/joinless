# SPDX-License-Identifier: MIT
"""The ``joinless`` console entry point: ``resolve``, ``compare``, ``doctor``.

RFC-0003 names five commands; only three exist at this point in the project (M1).
``report`` and ``benchmark`` are later work and deliberately absent here and from
``--help`` — this module's job is to name only the commands that actually exist,
not to reserve a place for ones that don't yet.

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

Every command here is local computation over :mod:`joinless.records`,
:mod:`joinless.resolver` and :mod:`joinless.scoring` plus stdlib file and platform
calls — nothing in this module ever opens a socket, so "the command completes with
no network interface available" (all three issues) holds structurally rather than by
a check this module performs.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import platform
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from joinless.records import Record
from joinless.resolver import ResolutionResult
from joinless.resolver import resolve as resolve_records
from joinless.scoring import ScorerUnavailable, ThresholdMatcher, get_scorer

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


_COMMANDS: Mapping[str, Callable[[argparse.Namespace], int]] = {
    "resolve": _cmd_resolve,
    "compare": _cmd_compare,
    "doctor": _cmd_doctor,
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

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command is None:
        return 0
    return _COMMANDS[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
