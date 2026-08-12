# SPDX-License-Identifier: MIT
"""RFC-0004 step 5: diff operator types between the fp32 and int8 graphs (issue #10).

Dynamic quantization converts what it is able to convert. If the encoder's matmuls come
through untouched, an int8 artefact that is smaller on disk is measuring compression,
not inference cost — this is the step that makes that visible rather than implied.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

MATMUL_FAMILY = ("MatMul", "Gemm")
"""The operator types :mod:`spikes.quantization.quantize_int8` requests conversion for."""

QUANTIZED_MATMUL_REPLACEMENTS = (
    "MatMulInteger",
    "QLinearMatMul",
    "QGemm",
    "DynamicQuantizeLinear",
)
"""Operator types ``quantize_dynamic`` introduces in place of a converted MatMul/Gemm."""


class NodeLike(Protocol):
    @property
    def op_type(self) -> str: ...


@dataclass(frozen=True, slots=True)
class OperatorDiff:
    added: tuple[str, ...]
    """Operator types present in the int8 graph, absent from the fp32 graph."""
    removed: tuple[str, ...]
    """Operator types present in the fp32 graph, absent from the int8 graph."""
    changed_counts: tuple[str, ...]
    """Operator types present in both graphs, at different counts."""


def enumerate_operator_types(nodes: Iterable[NodeLike]) -> Counter[str]:
    """Count nodes by ``op_type``. A pure function of the node list, so it is testable
    against a hand-built graph and needs no model fetch."""
    return Counter(node.op_type for node in nodes)


def load_operator_types(path: str | Path) -> Counter[str]:
    """Load an ``.onnx`` file and count its operator types."""
    import onnx

    model = onnx.load(str(path))
    return enumerate_operator_types(model.graph.node)


def diff_operator_types(
    fp32: Mapping[str, int], int8: Mapping[str, int]
) -> OperatorDiff:
    """Diff two operator-type counts as an explicit list, never a summary (issue #10)."""
    fp32_types = set(fp32)
    int8_types = set(int8)

    added = tuple(sorted(int8_types - fp32_types))
    removed = tuple(sorted(fp32_types - int8_types))
    changed_counts = tuple(
        sorted(op for op in fp32_types & int8_types if fp32[op] != int8[op])
    )
    return OperatorDiff(added=added, removed=removed, changed_counts=changed_counts)


def classify_matmul_conversion(
    fp32_counts: Mapping[str, int],
    int8_counts: Mapping[str, int],
    matmul_family: Sequence[str] = MATMUL_FAMILY,
) -> dict[str, object]:
    """Name which encoder matmuls were converted and which were not (issue #10)."""
    per_op: dict[str, object] = {}
    for op in matmul_family:
        fp32_count = fp32_counts.get(op, 0)
        int8_count = int8_counts.get(op, 0)
        per_op[op] = {
            "fp32_count": fp32_count,
            "int8_count_remaining": int8_count,
            "converted_count": max(fp32_count - int8_count, 0),
        }

    replacements_present = {
        op: int8_counts.get(op, 0)
        for op in QUANTIZED_MATMUL_REPLACEMENTS
        if int8_counts.get(op, 0) > 0
    }
    per_op["replacement_ops_present"] = replacements_present
    return per_op


def main(argv: list[str] | None = None) -> int:
    """Diff the fp32 and int8 graphs' operator types and record the result."""
    import argparse
    import os

    from spikes.quantization.cli_common import (
        read_fragment,
        resolve_cache_dir,
        write_fragment,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    cache_dir = resolve_cache_dir(os.environ)
    quantize_record = read_fragment(cache_dir, "step3_quantize")
    call = quantize_record["call"]
    assert isinstance(call, dict)
    fp32_path = str(call["model_input"])
    int8_path = str(call["model_output"])

    fp32_counts = load_operator_types(fp32_path)
    int8_counts = load_operator_types(int8_path)
    diff = diff_operator_types(fp32_counts, int8_counts)
    matmul_summary = classify_matmul_conversion(fp32_counts, int8_counts)

    write_fragment(
        cache_dir,
        "step5_operators",
        {
            "fp32_operator_counts": dict(fp32_counts),
            "int8_operator_counts": dict(int8_counts),
            "added": list(diff.added),
            "removed": list(diff.removed),
            "changed_counts": list(diff.changed_counts),
            "matmul_conversion": matmul_summary,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
