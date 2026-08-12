# SPDX-License-Identifier: MIT
"""Plumbing shared by every RFC-0004 step script.

Two things every step needs: where to put fetched artefacts and intermediate output
(the model cache directory, supplied by the maintainer rather than assumed to exist —
see ``spikes/quantization/README.md``), and a common shape for handing its findings to
the next step, or to the final record assembly (:mod:`spikes.quantization.record`).

The cache directory lives outside the repository. Nothing written under it is committed.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

CACHE_DIR_ENV_VAR = "JOINLESS_MODEL_CACHE_DIR"


class CacheDirNotSetError(RuntimeError):
    """Raised when the model cache directory environment variable is unset or empty."""


def resolve_cache_dir(environ: Mapping[str, str]) -> Path:
    """Read the model cache directory from ``environ`` rather than assuming a path.

    Takes the environment mapping as a parameter — never reads ``os.environ`` itself —
    so a caller can supply exactly one name and nothing else reaches this function.
    """
    value = environ.get(CACHE_DIR_ENV_VAR)
    if not value:
        raise CacheDirNotSetError(
            f"{CACHE_DIR_ENV_VAR} is not set. Point it at a writable directory for "
            "fetched model artefacts and intermediate spike output."
        )
    return Path(value)


def fragment_path(cache_dir: Path, name: str) -> Path:
    """The path a step's JSON fragment lives at, given its short name."""
    return cache_dir / f"{name}.json"


def write_fragment(cache_dir: Path, name: str, payload: Mapping[str, object]) -> Path:
    """Write one step's findings as JSON, creating the cache directory if needed."""
    path = fragment_path(cache_dir, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def read_fragment(cache_dir: Path, name: str) -> dict[str, object]:
    """Read a fragment a previous step wrote. Raises ``FileNotFoundError`` if absent —
    a missing prerequisite step is a reason to stop, not to invent a default."""
    path = fragment_path(cache_dir, name)
    result: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
    return result
