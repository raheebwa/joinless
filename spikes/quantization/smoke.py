# SPDX-License-Identifier: MIT
"""RFC-0004 step 6: fp32 vs int8 embedding divergence on a fixed smoke set (issue #11).

The smoke set below is authored directly for this spike — invented pairs, synthetic
throughout (ADR-0004), no real business or person. It is deliberately *not* described
as drawn from "the development split": the corpus
generator and its three-way split (ADR-0011) do not exist yet at this point in the
project (they are M1, issues #36-40), so there is nothing to draw from. What the split
exists to guarantee — that nothing calibration- or sealed-test-bound leaks into a tuning
decision — holds here for a stronger reason: no calibration or sealed-test data exists
yet for this smoke set to have touched.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NamePair:
    pair_id: str
    left: str
    right: str


SMOKE_SET: tuple[NamePair, ...] = (
    NamePair("p01", "Acme Trading Co", "Acme Trading Company"),
    NamePair("p02", "Nile Valley Traders", "Nile Vally Traders"),
    NamePair("p03", "Kampala Metal Works Ltd", "Kampala Metal Works Limited"),
    NamePair("p04", "Northgate Logistics", "North Gate Logistics"),
    NamePair("p05", "Silverline Textiles", "Silver Line Textiles"),
    NamePair("p06", "Blue Ridge Farmers Coop", "Blue Ridge Farmers Cooperative"),
    NamePair("p07", "Zenith Hardware Supplies", "Zenith Hardware Supply"),
    NamePair("p08", "Riverside Bakery", "River Side Bakery"),
    NamePair("p09", "Falcon Freight Services", "Falcon Frieght Services"),
    NamePair("p10", "Sunrise Agro Traders", "Sunrise Agro Trading"),
)
"""Ten invented near-duplicate name pairs. Every distinct name in either column is
embedded through both graphs; this step compares each name's fp32 embedding against its
own int8 embedding, not the two names in a pair against each other — that comparison is
the resolver's job, not this spike's."""


@dataclass(frozen=True, slots=True)
class NameDivergence:
    name: str
    cosine_similarity: float


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two equal-length vectors, stdlib only.

    Accepts any sequence of floats — a plain list in tests, a NumPy array at run time —
    so this function stays free of a NumPy import and is testable without one.
    """
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("cosine similarity is undefined for a zero vector")
    return dot / (norm_a * norm_b)


def mean_pool(
    token_embeddings: Sequence[Sequence[float]], attention_mask: Sequence[int]
) -> list[float]:
    """Mean-pool token embeddings over the unmasked positions only.

    Padding tokens carry an embedding but no meaning; including them would bias the
    pooled vector toward the padding scheme rather than the text.
    """
    dim = len(token_embeddings[0]) if token_embeddings else 0
    sums = [0.0] * dim
    count = 0
    for vector, mask in zip(token_embeddings, attention_mask, strict=True):
        if mask:
            for i, value in enumerate(vector):
                sums[i] += value
            count += 1
    if count == 0:
        raise ValueError("attention mask selects no tokens to pool")
    return [total / count for total in sums]


def build_divergence_report(
    divergences: Sequence[NameDivergence],
) -> dict[str, object]:
    """Assemble step 6's record: per-name similarity and the maximum divergence.

    ``max_divergence`` is ``None`` — never ``0.0`` — when the smoke set is empty
    (ADR-0013): an empty run and a run where every pair diverged maximally must not
    collapse to the same number.
    """
    similarities = [d.cosine_similarity for d in divergences]
    max_divergence = (1.0 - min(similarities)) if similarities else None
    return {
        "per_name": [
            {"name": d.name, "cosine_similarity": d.cosine_similarity}
            for d in divergences
        ],
        "max_divergence": max_divergence,
    }


def _embed(session: object, tokenizer: object, text: str) -> list[float]:
    """Tokenize, run one graph, and mean-pool one name's embedding.

    Untested directly: it drives a real tokenizer and a real ONNX Runtime session.
    :func:`mean_pool`, the logic it delegates to, is covered above.
    """
    encoded = tokenizer([text], padding=True, truncation=True, return_tensors="np")  # type: ignore[operator]
    feed = {
        name: encoded[name]
        for name in encoded
        if name in {"input_ids", "attention_mask", "token_type_ids"}
    }
    outputs = session.run(None, feed)  # type: ignore[attr-defined]
    token_embeddings = outputs[0][0]
    attention_mask = encoded["attention_mask"][0]
    return mean_pool(token_embeddings, attention_mask)


def main(argv: list[str] | None = None) -> int:
    """Embed every distinct smoke-set name through both graphs and record divergence."""
    import argparse
    import os

    import onnxruntime
    from transformers import AutoTokenizer

    from spikes.quantization.cli_common import (
        hf_cache_dir,
        read_fragment,
        resolve_cache_dir,
        write_fragment,
    )
    from spikes.quantization.model import MODEL_ID

    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    cache_dir = resolve_cache_dir(os.environ)
    quantize_record = read_fragment(cache_dir, "step3_quantize")
    call = quantize_record["call"]
    assert isinstance(call, dict)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, cache_dir=str(hf_cache_dir(cache_dir))
    )
    fp32_session = onnxruntime.InferenceSession(
        str(call["model_input"]), providers=["CPUExecutionProvider"]
    )
    int8_session = onnxruntime.InferenceSession(
        str(call["model_output"]), providers=["CPUExecutionProvider"]
    )

    names = sorted(
        {pair.left for pair in SMOKE_SET} | {pair.right for pair in SMOKE_SET}
    )
    divergences = []
    for name in names:
        fp32_embedding = _embed(fp32_session, tokenizer, name)
        int8_embedding = _embed(int8_session, tokenizer, name)
        divergences.append(
            NameDivergence(
                name=name,
                cosine_similarity=cosine_similarity(fp32_embedding, int8_embedding),
            )
        )

    report = build_divergence_report(divergences)
    write_fragment(
        cache_dir,
        "step6_smoke",
        {
            "smoke_set_size": len(SMOKE_SET),
            "provenance": "authored for this spike; not drawn from a corpus split (none exists yet)",
            **report,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
