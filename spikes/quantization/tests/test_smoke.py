# SPDX-License-Identifier: MIT
"""RFC-0004 step 6: fp32 vs int8 embedding divergence on a fixed smoke set (issue #11)."""

from __future__ import annotations

import math

import pytest

from spikes.quantization.smoke import (
    SMOKE_SET,
    NameDivergence,
    build_divergence_report,
    cosine_similarity,
    mean_pool,
)


def test_smoke_set_is_invented_and_nonempty() -> None:
    assert len(SMOKE_SET) >= 8
    names = {pair.left for pair in SMOKE_SET} | {pair.right for pair in SMOKE_SET}
    # Synthetic fixtures only (ADR-0004): nothing resembling a phone number or a real
    # registration identifier belongs in a name string.
    assert not any(char.isdigit() for name in names for char in name)


def test_cosine_similarity_of_identical_vectors_is_one() -> None:
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_similarity_of_orthogonal_vectors_is_zero() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_of_opposite_vectors_is_minus_one() -> None:
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_similarity_rejects_a_zero_vector() -> None:
    with pytest.raises(ValueError, match="zero vector"):
        cosine_similarity([0.0, 0.0], [1.0, 0.0])


def test_mean_pool_averages_only_unmasked_tokens() -> None:
    token_embeddings = [[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]]
    attention_mask = [1, 1, 0]

    pooled = mean_pool(token_embeddings, attention_mask)

    assert pooled == pytest.approx([2.0, 2.0])


def test_mean_pool_rejects_an_all_zero_mask() -> None:
    with pytest.raises(ValueError, match="no tokens"):
        mean_pool([[1.0, 1.0]], [0])


def test_build_divergence_report_records_max_divergence() -> None:
    divergences = [
        NameDivergence(name="a", cosine_similarity=1.0),
        NameDivergence(name="b", cosine_similarity=0.9),
    ]

    report = build_divergence_report(divergences)
    per_name = report["per_name"]
    assert isinstance(per_name, list)

    assert report["max_divergence"] == pytest.approx(0.1)
    assert per_name[1]["name"] == "b"


def test_build_divergence_report_max_divergence_is_null_when_empty() -> None:
    report = build_divergence_report([])

    assert report["max_divergence"] is None
    assert report["per_name"] == []


def test_build_divergence_report_matches_manual_computation() -> None:
    divergences = [
        NameDivergence(name="a", cosine_similarity=0.995),
        NameDivergence(name="b", cosine_similarity=0.87),
        NameDivergence(name="c", cosine_similarity=0.999),
    ]

    report = build_divergence_report(divergences)
    max_divergence = report["max_divergence"]
    assert isinstance(max_divergence, float)

    assert max_divergence == pytest.approx(1 - 0.87)
    assert math.isclose(max_divergence, 0.13, rel_tol=1e-9)
