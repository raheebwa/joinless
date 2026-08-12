# SPDX-License-Identifier: MIT
"""RFC-0004 step 5: diff operator types between the fp32 and int8 graphs (issue #10).

The end-to-end case runs the real ``quantize_dynamic`` over a hand-built two-MatMul
graph — not the spike's model, no network — so the diff is checked against what the
library actually converts.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnxruntime.quantization import quantize_dynamic

from spikes.quantization.operators import (
    MATMUL_FAMILY,
    classify_matmul_conversion,
    diff_operator_types,
    enumerate_operator_types,
    load_operator_types,
)


def _toy_two_matmul_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])
    w1 = np.linspace(-1.0, 1.0, num=16, dtype=np.float32).reshape(4, 4)
    w2 = np.linspace(1.0, -1.0, num=16, dtype=np.float32).reshape(4, 4)
    w1_init = helper.make_tensor(
        "W1", TensorProto.FLOAT, w1.shape, w1.flatten().tolist()
    )
    w2_init = helper.make_tensor(
        "W2", TensorProto.FLOAT, w2.shape, w2.flatten().tolist()
    )
    node1 = helper.make_node("MatMul", ["X", "W1"], ["H"])
    node2 = helper.make_node("MatMul", ["H", "W2"], ["Y"])
    graph = helper.make_graph(
        [node1, node2], "toy-two-matmul", [x], [y], initializer=[w1_init, w2_init]
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])


def test_enumerate_operator_types_counts_by_op_type() -> None:
    model = _toy_two_matmul_model()

    counts = enumerate_operator_types(model.graph.node)

    assert counts == Counter({"MatMul": 2})


def test_load_operator_types_reads_from_disk(tmp_path) -> None:
    path = tmp_path / "m.onnx"
    onnx.save(_toy_two_matmul_model(), str(path))

    assert load_operator_types(path) == Counter({"MatMul": 2})


def test_diff_operator_types_reports_added_removed_and_changed() -> None:
    fp32 = Counter({"MatMul": 2, "Add": 1})
    int8 = Counter({"MatMulInteger": 2, "Add": 1, "DynamicQuantizeLinear": 1})

    diff = diff_operator_types(fp32, int8)

    assert diff.added == ("DynamicQuantizeLinear", "MatMulInteger")
    assert diff.removed == ("MatMul",)
    assert diff.changed_counts == ()  # "Add" present both sides at equal count


def test_diff_operator_types_reports_a_count_change_for_a_shared_op() -> None:
    fp32 = Counter({"MatMul": 2})
    int8 = Counter({"MatMul": 1, "MatMulInteger": 1})

    diff = diff_operator_types(fp32, int8)

    assert "MatMul" in diff.changed_counts


def test_classify_matmul_conversion_end_to_end(tmp_path) -> None:
    fp32_path = tmp_path / "fp32.onnx"
    int8_path = tmp_path / "int8.onnx"
    onnx.save(_toy_two_matmul_model(), str(fp32_path))
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        op_types_to_quantize=list(MATMUL_FAMILY),
    )

    fp32_counts = load_operator_types(fp32_path)
    int8_counts = load_operator_types(int8_path)
    summary = classify_matmul_conversion(fp32_counts, int8_counts)
    matmul_summary = summary["MatMul"]
    assert isinstance(matmul_summary, dict)

    assert matmul_summary["fp32_count"] == 2
    assert matmul_summary["converted_count"] == 2
    assert matmul_summary["int8_count_remaining"] == 0
    assert summary["replacement_ops_present"]  # something replaced the MatMuls
