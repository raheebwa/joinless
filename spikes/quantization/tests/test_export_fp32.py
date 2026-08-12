# SPDX-License-Identifier: MIT
"""RFC-0004 step 2: export the fp32 ONNX graph (issue #7).

Pure logic only: building the export command and capturing tool versions. The actual
subprocess invocation is a thin, untested pass-through — see the module docstring.
"""

from __future__ import annotations

from pathlib import Path

from spikes.quantization.export_fp32 import (
    EXPORT_TOOL_PACKAGES,
    build_export_command,
    capture_tool_versions,
)


def test_build_export_command_names_the_pinned_snapshot_task_and_output() -> None:
    """The model is the local snapshot step 1 pinned, not the hub id: `optimum-cli
    export onnx` has no --revision flag, so naming the hub id would export whatever
    HEAD happens to be — a different model from the one step 1 recorded."""
    command = build_export_command(
        model_path="/cache/hf/snapshots/deadbeef",
        output_dir=Path("/cache/fp32"),
    )

    assert command == [
        "optimum-cli",
        "export",
        "onnx",
        "--model",
        "/cache/hf/snapshots/deadbeef",
        "--task",
        "feature-extraction",
        "/cache/fp32",
    ]
    assert "--revision" not in command


def test_build_export_command_accepts_a_different_task() -> None:
    command = build_export_command(
        model_path="/snap",
        output_dir=Path("/o"),
        task="sentence-similarity",
    )

    assert "--task" in command
    assert command[command.index("--task") + 1] == "sentence-similarity"


def test_capture_tool_versions_uses_only_named_packages() -> None:
    versions = capture_tool_versions(get_version=lambda name: f"{name}-9.9")

    assert set(versions) == set(EXPORT_TOOL_PACKAGES)
    assert versions["torch"] == "torch-9.9"


def test_capture_tool_versions_omits_a_package_that_is_not_installed() -> None:
    def get_version(name: str) -> str | None:
        return None if name == "torch" else "1.0"

    versions = capture_tool_versions(get_version=get_version)

    assert "torch" not in versions
    assert versions["onnx"] == "1.0"


def test_build_export_command_routes_the_fetch_to_the_cache_dir() -> None:
    """Step 1 fetches into the supplied cache directory; step 2 shells out to a tool
    with its own default cache. Without --cache_dir the weights are fetched a second
    time, outside the directory the maintainer supplied and nominated as the whole
    footprint of the run."""
    command = build_export_command(
        "/local/snapshot",
        Path("/tmp/out"),
        cache_dir=Path("/tmp/cache/hf"),
    )
    assert "--cache_dir" in command
    assert command[command.index("--cache_dir") + 1] == "/tmp/cache/hf"
