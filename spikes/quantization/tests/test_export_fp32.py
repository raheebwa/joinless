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


def test_build_export_command_names_model_revision_task_and_output() -> None:
    command = build_export_command(
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        revision="deadbeef",
        output_dir=Path("/cache/fp32"),
    )

    assert command == [
        "optimum-cli",
        "export",
        "onnx",
        "--model",
        "sentence-transformers/all-MiniLM-L6-v2",
        "--revision",
        "deadbeef",
        "--task",
        "feature-extraction",
        "/cache/fp32",
    ]


def test_build_export_command_accepts_a_different_task() -> None:
    command = build_export_command(
        model_id="m",
        revision="r",
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
