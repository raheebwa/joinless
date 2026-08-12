# SPDX-License-Identifier: MIT
"""Shared plumbing: locating the model cache directory, and the JSON fragments each
RFC-0004 step contributes toward the final spike record.
"""

from __future__ import annotations

import pytest

from spikes.quantization.cli_common import (
    CACHE_DIR_ENV_VAR,
    CacheDirNotSetError,
    hf_cache_dir,
    read_fragment,
    resolve_cache_dir,
    write_fragment,
)


def test_resolve_cache_dir_reads_the_named_variable(tmp_path) -> None:
    environ = {CACHE_DIR_ENV_VAR: str(tmp_path)}

    assert resolve_cache_dir(environ) == tmp_path


def test_resolve_cache_dir_raises_when_unset() -> None:
    with pytest.raises(CacheDirNotSetError, match=CACHE_DIR_ENV_VAR):
        resolve_cache_dir({})


def test_resolve_cache_dir_raises_when_empty() -> None:
    with pytest.raises(CacheDirNotSetError):
        resolve_cache_dir({CACHE_DIR_ENV_VAR: ""})


def test_write_then_read_fragment_round_trips(tmp_path) -> None:
    payload = {"step": "export", "command": ["optimum-cli", "export", "onnx"]}

    written = write_fragment(tmp_path, "step2_export", payload)

    assert written.exists()
    assert read_fragment(tmp_path, "step2_export") == payload


def test_read_fragment_missing_file_raises_file_not_found(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        read_fragment(tmp_path, "does_not_exist")


def test_hf_cache_dir_is_a_named_subdirectory_of_the_cache_dir(tmp_path) -> None:
    assert hf_cache_dir(tmp_path) == tmp_path / "hf"
