# SPDX-License-Identifier: MIT
"""RFC-0004 step 3: produce the int8 graph via quantize_dynamic (issue #8).

Exercises the real ``onnxruntime.quantization.quantize_dynamic`` against a tiny,
hand-built ONNX graph — not the spike's target model, no network, no download — so the
parameters this module passes are checked against the library's actual behaviour rather
than a mocked stand-in.
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnxruntime.quantization import QuantType, quantize_dynamic

from spikes.quantization.quantize_int8 import (
    OP_TYPES_TO_QUANTIZE,
    build_quantize_kwargs,
    run_quantization,
    serialize_quantize_kwargs,
)


def _toy_matmul_model() -> onnx.ModelProto:
    """A single MatMul against a constant weight — the shape dynamic quantization's
    default ``MatMulConstBOnly`` requires to convert a node at all."""
    a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])
    weight = np.linspace(-1.0, 1.0, num=16, dtype=np.float32).reshape(4, 4)
    weight_initializer = helper.make_tensor(
        "B", TensorProto.FLOAT, weight.shape, weight.flatten().tolist()
    )
    node = helper.make_node("MatMul", ["A", "B"], ["Y"])
    graph = helper.make_graph(
        [node], "toy-matmul", [a], [y], initializer=[weight_initializer]
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])


def test_build_quantize_kwargs_names_every_parameter(tmp_path) -> None:
    kwargs = build_quantize_kwargs(tmp_path / "fp32.onnx", tmp_path / "int8.onnx")

    assert kwargs["model_input"] == str(tmp_path / "fp32.onnx")
    assert kwargs["model_output"] == str(tmp_path / "int8.onnx")
    assert kwargs["weight_type"] == QuantType.QInt8
    assert kwargs["per_channel"] is False
    assert kwargs["reduce_range"] is False
    assert kwargs["op_types_to_quantize"] == list(OP_TYPES_TO_QUANTIZE)
    assert kwargs["extra_options"] == {}


def test_serialize_quantize_kwargs_makes_weight_type_json_safe(tmp_path) -> None:
    kwargs = build_quantize_kwargs(tmp_path / "fp32.onnx", tmp_path / "int8.onnx")

    serialized = serialize_quantize_kwargs(kwargs)

    assert serialized["weight_type"] == str(QuantType.QInt8)
    assert isinstance(serialized["weight_type"], str)
    assert serialized["op_types_to_quantize"] == list(OP_TYPES_TO_QUANTIZE)


def test_run_quantization_converts_the_matmul(tmp_path) -> None:
    input_path = tmp_path / "fp32.onnx"
    output_path = tmp_path / "int8.onnx"
    onnx.save(_toy_matmul_model(), str(input_path))
    kwargs = build_quantize_kwargs(input_path, output_path)

    run_quantization(kwargs, quantize_dynamic)

    quantized = onnx.load(str(output_path))
    op_types = {node.op_type for node in quantized.graph.node}
    assert "MatMul" not in op_types
    assert op_types & {"MatMulInteger", "DynamicQuantizeLinear"}


def test_run_quantization_passes_kwargs_through_verbatim() -> None:
    calls: list[dict[str, object]] = []

    run_quantization(
        {"model_input": "a", "model_output": "b"}, lambda **kw: calls.append(kw)
    )

    assert calls == [{"model_input": "a", "model_output": "b"}]
