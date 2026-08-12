# SPDX-License-Identifier: MIT
"""RFC-0004 step 1: select one model and record its identity, revision, checksum and
licence (issue #7).

Model selection is a separate question from this spike (PRD NG5) — one model only, a
stock sentence encoder small enough to be a plausible client-device candidate.

``MODEL_ID`` is the only fact declared here. The revision (resolved commit SHA) and the
licence come from the model host at run time via :func:`resolve_model_selection`, not
from a value written into source: a licence string typed from memory into this file could
be stale or wrong, and the whole point of recording it is that it traces to what the
model card actually said at fetch time.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
"""The one model this spike measures.

A 22M-parameter stock sentence encoder — small enough that ADR-0009 names this exact
parameter count when arguing a model of this size may already be resident in cache and
bound by something other than weight bandwidth, which is precisely the question this
spike exists to check rather than assume.
"""

UNRESOLVED_LICENSE = "unresolved-see-model-card"
"""Recorded when neither ``cardData.license`` nor a ``license:`` tag is present.

Never substituted with a guessed licence string — an unresolved licence is a fact about
the model card, not a defect in this function.
"""


class ModelInfoLike(Protocol):
    """The subset of ``huggingface_hub.hf_api.ModelInfo`` this step reads.

    A protocol rather than an import of the real type, so the resolution logic below is
    testable with a plain stand-in and never has to construct a real API response.
    """

    @property
    def sha(self) -> str | None: ...

    @property
    def card_data(self) -> Mapping[str, object] | None: ...

    @property
    def tags(self) -> Sequence[str]: ...


@dataclass(frozen=True, slots=True)
class PlainModelInfo:
    """A plain, dependency-free ``ModelInfoLike``.

    Used two ways: as a fixture in tests, and in :func:`main` to adapt the real
    ``huggingface_hub`` response — whose ``card_data`` is a ``ModelCardData`` object and
    whose ``tags`` may be ``None`` — down to the shape :func:`resolve_model_selection`
    actually needs.
    """

    sha: str | None
    card_data: Mapping[str, object] | None
    tags: Sequence[str]


@dataclass(frozen=True, slots=True)
class ModelSelection:
    """Step 1's recorded outcome: what RFC-0004 calls "identity, revision ... licence"."""

    model_id: str
    revision: str
    license: str
    license_source: str


def resolve_license(
    card_data: Mapping[str, object] | None, tags: Sequence[str]
) -> tuple[str, str]:
    """Return ``(license, source)`` from a model card, preferring the structured field.

    ``cardData.license`` is the field the Hub itself renders as the licence badge.
    A ``license:...`` tag is the fallback some repositories carry instead.
    """
    if card_data is not None:
        license_value = card_data.get("license")
        if isinstance(license_value, str) and license_value:
            return license_value, "cardData.license"

    for tag in tags:
        prefix = "license:"
        if tag.startswith(prefix):
            candidate = tag[len(prefix) :]
            if candidate:
                return candidate, "tag"

    return UNRESOLVED_LICENSE, "unresolved"


def resolve_model_selection(info: ModelInfoLike) -> ModelSelection:
    """Assemble step 1's record from a model host's response for :data:`MODEL_ID`.

    Raises ``ValueError`` rather than recording a placeholder revision: a selection
    without a resolved commit cannot be re-fetched deterministically, which defeats the
    reason step 1 exists.
    """
    if not info.sha:
        raise ValueError("model host did not return a resolved commit sha")

    license_value, license_source = resolve_license(info.card_data, info.tags)
    return ModelSelection(
        model_id=MODEL_ID,
        revision=info.sha,
        license=license_value,
        license_source=license_source,
    )


WEIGHTS_FILE_PRIORITY = ("model.safetensors", "pytorch_model.bin")
"""Candidate weights filenames, in the order this step prefers them.

``safetensors`` first: it is the format the model card's own widget uses and it carries
no arbitrary-code-execution risk on load, unlike a pickle-based ``.bin``.
"""


def select_weights_file(available: Sequence[str]) -> str:
    """Pick the weights file to checksum out of a repository's file listing."""
    for candidate in WEIGHTS_FILE_PRIORITY:
        if candidate in available:
            return candidate
    raise ValueError(f"none of {WEIGHTS_FILE_PRIORITY} found in {sorted(available)}")


def sha256_file(path: str | Path) -> str:
    """Checksum a fetched artefact so the record names exactly what was measured."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    """Resolve :data:`MODEL_ID` against the live model host and record step 1.

    Untested: the network call to the model host and the artefact download. Everything
    it feeds into (:func:`resolve_model_selection`, :func:`select_weights_file`,
    :func:`sha256_file`) is pure and covered above.
    """
    import argparse
    import os

    from huggingface_hub import HfApi, snapshot_download

    from spikes.quantization.cli_common import (
        hf_cache_dir,
        resolve_cache_dir,
        write_fragment,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--revision", default=None, help="pin a specific commit; default HEAD"
    )
    args = parser.parse_args(argv)

    cache_dir = resolve_cache_dir(os.environ)
    raw_info = HfApi().model_info(MODEL_ID, revision=args.revision, expand=["cardData"])
    card_data = raw_info.card_data.to_dict() if raw_info.card_data is not None else None
    selection = resolve_model_selection(
        PlainModelInfo(sha=raw_info.sha, card_data=card_data, tags=raw_info.tags or [])
    )

    local_dir = snapshot_download(
        repo_id=MODEL_ID,
        revision=selection.revision,
        cache_dir=str(hf_cache_dir(cache_dir)),
    )
    available = os.listdir(local_dir)
    weights_file = select_weights_file(available)
    checksum = sha256_file(Path(local_dir) / weights_file)

    write_fragment(
        cache_dir,
        "step1_model",
        {
            "model_id": selection.model_id,
            "revision": selection.revision,
            "license": selection.license,
            "license_source": selection.license_source,
            "weights_file": weights_file,
            "checksum_sha256": checksum,
            "local_dir": local_dir,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
