# SPDX-License-Identifier: MIT
"""RFC-0004 step 1: select and identify one model.

Pure logic only. The network call that resolves a real model's metadata is not
exercised here — these tests supply fixed stand-ins for what that call returns.
"""

from __future__ import annotations

import hashlib

import pytest

from spikes.quantization.model import (
    UNRESOLVED_LICENSE,
    WEIGHTS_FILE_PRIORITY,
    PlainModelInfo,
    resolve_license,
    resolve_model_selection,
    select_weights_file,
    sha256_file,
)


def test_resolve_license_prefers_card_data() -> None:
    license_value, source = resolve_license(
        card_data={"license": "apache-2.0"}, tags=["license:mit"]
    )

    assert license_value == "apache-2.0"
    assert source == "cardData.license"


def test_resolve_license_falls_back_to_tag() -> None:
    license_value, source = resolve_license(
        card_data=None, tags=["pytorch", "license:mit"]
    )

    assert license_value == "mit"
    assert source == "tag"


def test_resolve_license_unresolved_when_absent() -> None:
    license_value, source = resolve_license(card_data={}, tags=["pytorch"])

    assert license_value == UNRESOLVED_LICENSE
    assert source == "unresolved"


def test_resolve_model_selection_combines_revision_and_license() -> None:
    info = PlainModelInfo(sha="abc123", card_data={"license": "apache-2.0"}, tags=[])

    selection = resolve_model_selection(info)

    assert selection.revision == "abc123"
    assert selection.license == "apache-2.0"
    assert selection.license_source == "cardData.license"
    assert selection.model_id


def test_resolve_model_selection_rejects_unresolved_revision() -> None:
    info = PlainModelInfo(sha=None, card_data=None, tags=[])

    with pytest.raises(ValueError, match="resolved commit sha"):
        resolve_model_selection(info)


def test_select_weights_file_prefers_safetensors() -> None:
    chosen = select_weights_file(
        ["config.json", "pytorch_model.bin", "model.safetensors"]
    )

    assert chosen == WEIGHTS_FILE_PRIORITY[0] == "model.safetensors"


def test_select_weights_file_falls_back_to_pytorch_bin() -> None:
    chosen = select_weights_file(["config.json", "pytorch_model.bin"])

    assert chosen == "pytorch_model.bin"


def test_select_weights_file_raises_when_none_present() -> None:
    with pytest.raises(ValueError, match="none of"):
        select_weights_file(["config.json", "tokenizer.json"])


def test_sha256_file_matches_hashlib(tmp_path) -> None:
    target = tmp_path / "weights.bin"
    payload = b"synthetic weights, not a real model artefact"
    target.write_bytes(payload)

    assert sha256_file(target) == hashlib.sha256(payload).hexdigest()


def test_model_info_fields_request_the_commit_sha() -> None:
    """The model host returns only the fields named in `expand`. Omitting "sha" makes
    `info.sha` None, which resolve_model_selection correctly refuses — so step 1 fails
    with no revision rather than recording a wrong one. The revision is the whole point
    of step 1, so it has to be asked for."""
    from spikes.quantization.model import MODEL_INFO_FIELDS

    assert "sha" in MODEL_INFO_FIELDS
    assert "cardData" in MODEL_INFO_FIELDS
