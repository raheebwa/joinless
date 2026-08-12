# SPDX-License-Identifier: MIT
"""RFC-0004 step 2: export the fp32 ONNX graph, fully scripted and re-runnable
(issue #7).

``build_export_command`` and ``capture_tool_versions`` are pure and tested against
fixtures. ``main`` is the thin, untested wrapper that actually shells out to
``optimum-cli`` and writes the record fragment step 8 reads — untestable without the
export toolchain installed and a real model fetch, which is precisely the boundary the
project's TDD policy asks to be named rather than hidden.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import subprocess
from collections.abc import Callable
from pathlib import Path

from spikes.quantization.cli_common import (
    hf_cache_dir,
    read_fragment,
    resolve_cache_dir,
    write_fragment,
)

EXPORT_TOOL_PACKAGES = ("torch", "transformers", "optimum", "onnx")
"""The export-time tools whose versions the record names (RFC-0004 step 2).

An explicit list, not a dump of every installed distribution — the record states what
took part in producing the artefact, not what happens to be on the machine.
"""

DEFAULT_TASK = "feature-extraction"


def build_export_command(
    model_path: str,
    output_dir: Path,
    *,
    task: str = DEFAULT_TASK,
    cache_dir: Path | None = None,
) -> list[str]:
    """The exact ``optimum-cli`` invocation this step records and runs.

    ``cache_dir`` is passed through because this step shells out to a tool with its
    own default cache. Without it the weights step 1 already fetched are fetched a
    second time, outside the directory the maintainer supplied — so deleting that
    directory afterwards would not undo what the run put on the machine.
    """
    command = [
        "optimum-cli",
        "export",
        "onnx",
        "--model",
        model_path,
        "--task",
        task,
    ]
    if cache_dir is not None:
        command += ["--cache_dir", str(cache_dir)]
    command.append(str(output_dir))
    return command


def capture_tool_versions(
    get_version: Callable[[str], str | None],
) -> dict[str, str]:
    """Resolve a version per name in :data:`EXPORT_TOOL_PACKAGES`, via an injected
    lookup so this is testable without the packages being installed."""
    versions: dict[str, str] = {}
    for name in EXPORT_TOOL_PACKAGES:
        version = get_version(name)
        if version is not None:
            versions[name] = version
    return versions


def _get_installed_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=DEFAULT_TASK)
    args = parser.parse_args(argv)

    cache_dir = resolve_cache_dir(os.environ)
    selection = read_fragment(cache_dir, "step1_model")
    output_dir = cache_dir / "fp32"

    command = build_export_command(
        # Export from the snapshot step 1 already pinned and fetched. `optimum-cli
        # export onnx` has no --revision flag, and pointing it at the hub id would
        # export whatever HEAD is now — a different model from the one recorded.
        model_path=str(selection["local_dir"]),
        output_dir=output_dir,
        task=args.task,
        cache_dir=hf_cache_dir(cache_dir),
    )
    subprocess.run(command, check=True)

    write_fragment(
        cache_dir,
        "step2_export",
        {
            "command": command,
            "output_dir": str(output_dir),
            "tool_versions": capture_tool_versions(_get_installed_version),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
