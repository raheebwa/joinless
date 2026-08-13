# SPDX-License-Identifier: MIT
"""Generate the README's results section from a run record (issue #72).

    uv run python scripts/render_readme_results.py <record> [readme]

Reads one JSON run record written by ``joinless benchmark``, renders it with
:func:`joinless.readme_results.render_results_section`, and splices the
result into ``readme`` (``README.md`` by default) between that module's
``MARKER_BEGIN``/``MARKER_END`` pair, in place.

**Why a script under a committed path, not a sixth `joinless` subcommand.**
RFC-0003 fixes the CLI at exactly five commands — ``resolve``, ``compare``,
``benchmark``, ``report``, ``doctor`` — each one a reproduction primitive a
reader runs directly against their own data or their own run. This tool is a
different kind of thing: it always targets one already-written, already
canonical record and always writes to one fixed file, ``README.md``, never to
stdout and never to a path a reader supplies for their own purpose. That is
an authoring step in this repository's own documentation, not a measurement
or reproduction primitive a reader runs against arbitrary input the way
``report`` renders an arbitrary record path to a terminal. Adding it to the
five-command surface would blur the boundary that RFC-0003 draws on purpose;
a script under a committed path keeps the CLI's contract unchanged while
still being a command any reader can run, read, and check against the
committed record it names (issue #72's own "generated ... by a committed
command" — the requirement is that the command is committed and reproducible,
not that it lives behind ``joinless``'s own argument parser).

All of the logic that decides what the section says lives in
:mod:`joinless.readme_results`, under the project's ordinary 100%-coverage
floor — this script is deliberately the thin, untested part: file reads, one
function call, a file write, mirroring how ``joinless.cli._cmd_report``
keeps the same split between a covered rendering module and an I/O shell
around it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from joinless.readme_results import render_results_section, splice_into_readme


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "record", type=Path, help="Path to a benchmarks/*.json run record."
    )
    parser.add_argument(
        "readme",
        type=Path,
        nargs="?",
        default=Path("README.md"),
        help="README to splice the generated section into (default: README.md).",
    )
    args = parser.parse_args(argv)

    try:
        with args.record.open(encoding="utf-8") as handle:
            record = json.load(handle)
    except OSError as exc:
        print(f"could not read {args.record}: {exc}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"{args.record} is not a valid JSON run record: {exc}", file=sys.stderr)
        return 1

    section = render_results_section(record)

    try:
        readme_text = args.readme.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"could not read {args.readme}: {exc}", file=sys.stderr)
        return 1

    updated = splice_into_readme(readme_text, section)
    args.readme.write_text(updated, encoding="utf-8")
    print(f"wrote the results section generated from {args.record} into {args.readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
