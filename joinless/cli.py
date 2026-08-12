# SPDX-License-Identifier: MIT
"""The ``joinless`` console entry point.

No subcommands are declared here: none exist yet, and ``--help`` must name
only the commands that actually exist at this point in the project.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        prog="joinless",
        description="Keyless entity resolution: measurement and reproduction surface.",
    )


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
