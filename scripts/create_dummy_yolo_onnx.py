#!/usr/bin/env python3
"""Generate a minimal deterministic YOLOv8-shaped ONNX model (models/dummy_yolo.onnx).

Graph specification:
  - Input:  "images"  float32 [1, 3, 640, 640]
  - Output: "output0" float32 [1, 7, 8400]  (4 box coords + 3 safety classes)
  - Anchor #0 emits a synthetic high-confidence Person detection:
      cx=320.0, cy=320.0, w=100.0, h=200.0, score(Person=0)=0.95

Uses `onnx.helper` when the `onnx` package is available, with a zero-dependency
pure-Python ONNX Protobuf wire serializer fallback so unit tests can generate the
model in minimal CI/embedded environments without external Python packages.
"""

from __future__ import annotations

import array
from pathlib import Path
import sys


NUM_CLASSES = 3
NUM_CHANNELS = 4 + NUM_CLASSES  # 7: [cx, cy, w, h, Person, Helmet, Vest]
NUM_ANCHORS = 8400


def _build_prediction_buffer() -> bytes:
    """Create raw little-endian float32 bytes for shape [1, 7, 8400]."""
    values = array.array("f", [0.0]) * (NUM_CHANNELS * NUM_ANCHORS)
    # Populate anchor #0 across channel rows (row-major [7, 8400])
    values[0 * NUM_ANCHORS + 0] = 320.0  # cx
    values[1 * NUM_ANCHORS + 0] = 320.0  # cy
    values[2 * NUM_ANCHORS + 0] = 100.0  # w
    values[3 * NUM_ANCHORS + 0] = 200.0  # h
    values[4 * NUM_ANCHORS + 0] = 0.95   # class 0: Person
    values[5 * NUM_ANCHORS + 0] = 0.05   # class 1: Helmet
    values[6 * NUM_ANCHORS + 0] = 0.02   # class 2: Vest

    if sys.byteorder != "little":
        values.byteswap()
    return values.tobytes()


def _create_with_onnx_package(output_path: Path, pred_bytes: bytes) -> None:
    import onnx
    from onnx import TensorProto, checker, helper

    images_in = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])
    output_out = helper.make_tensor_value_info(
        "output0", TensorProto.FLOAT, [1, NUM_CHANNELS, NUM_ANCHORS]
    )

    zero_scalar = helper.make_tensor(
        name="zero_scalar",
        data_type=TensorProto.FLOAT,
        dims=[],
        vals=[0.0],
    )
    dummy_pred = helper.make_tensor(
        name="dummy_pred",
        data_type=TensorProto.FLOAT,
        dims=[1, NUM_CHANNELS, NUM_ANCHORS],
        vals=pred_bytes,
        raw=True,
    )

    reduce_node = helper.make_node(
        "ReduceMean", inputs=["images"], outputs=["img_mean"], name="reduce_images", keepdims=0
    )
    mul_node = helper.make_node(
        "Mul", inputs=["img_mean", "zero_scalar"], outputs=["zero_dep"], name="zero_mul"
    )
    add_node = helper.make_node(
        "Add", inputs=["dummy_pred", "zero_dep"], outputs=["output0"], name="emit_output"
    )

    graph = helper.make_graph(
        nodes=[reduce_node, mul_node, add_node],
        name="DummyYoloGraph",
        inputs=[images_in],
        outputs=[output_out],
        initializer=[zero_scalar, dummy_pred],
    )
    model = helper.make_model(
        graph,
        producer_name="safety_engine_test_generator",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 7
    checker.check_model(model)
    onnx.save(model, str(output_path))


# ------------------------------------------------------------------------------
# Minimal Protobuf wire-format helpers for zero-dependency ONNX serialization
# ------------------------------------------------------------------------------
def _encode_varint(value: int) -> bytes:
    out = bytearray()
    while value > 0x7F:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value & 0x7F)
    return bytes(out)


def _field_varint(field_num: int, value: int) -> bytes:
    return _encode_varint((field_num << 3) | 0) + _encode_varint(value)


def _field_bytes(field_num: int, payload: bytes) -> bytes:
    return _encode_varint((field_num << 3) | 2) + _encode_varint(len(payload)) + payload


def _field_str(field_num: int, text: str) -> bytes:
    return _field_bytes(field_num, text.encode("utf-8"))


def _make_value_info_proto(name: str, elem_type: int, dims: list[int]) -> bytes:
    dim_msgs = b"".join(_field_bytes(1, _field_varint(1, d)) for d in dims)
    tensor_type_msg = _field_varint(1, elem_type) + _field_bytes(2, dim_msgs)
    type_proto_msg = _field_bytes(1, tensor_type_msg)
    return _field_str(1, name) + _field_bytes(2, type_proto_msg)


def _make_tensor_proto(name: str, elem_type: int, dims: list[int], raw_data: bytes) -> bytes:
    msg = b"".join(_field_varint(1, d) for d in dims)
    msg += _field_varint(2, elem_type)
    msg += _field_str(8, name)
    msg += _field_bytes(9, raw_data)
    return msg


def _make_attr_int(name: str, value: int) -> bytes:
    # AttributeProto: name=1, i=3, type=20 (INT=2)
    return _field_str(1, name) + _field_varint(3, value) + _field_varint(20, 2)


def _make_node_proto(
    op_type: str,
    inputs: list[str],
    outputs: list[str],
    name: str,
    attrs: list[bytes] | None = None,
) -> bytes:
    msg = b"".join(_field_str(1, inp) for inp in inputs)
    msg += b"".join(_field_str(2, out) for out in outputs)
    msg += _field_str(3, name)
    msg += _field_str(4, op_type)
    for attr in attrs or []:
        msg += _field_bytes(5, attr)
    return msg


def _create_with_wire_serializer(output_path: Path, pred_bytes: bytes) -> None:
    float_type = 1  # TensorProto.FLOAT

    images_vi = _make_value_info_proto("images", float_type, [1, 3, 640, 640])
    output_vi = _make_value_info_proto("output0", float_type, [1, NUM_CHANNELS, NUM_ANCHORS])

    zero_raw = array.array("f", [0.0])
    if sys.byteorder != "little":
        zero_raw.byteswap()
    zero_init = _make_tensor_proto("zero_scalar", float_type, [], zero_raw.tobytes())
    pred_init = _make_tensor_proto(
        "dummy_pred", float_type, [1, NUM_CHANNELS, NUM_ANCHORS], pred_bytes
    )

    n_reduce = _make_node_proto(
        "ReduceMean", ["images"], ["img_mean"], "reduce_images", [_make_attr_int("keepdims", 0)]
    )
    n_mul = _make_node_proto("Mul", ["img_mean", "zero_scalar"], ["zero_dep"], "zero_mul")
    n_add = _make_node_proto("Add", ["dummy_pred", "zero_dep"], ["output0"], "emit_output")

    graph_msg = (
        _field_bytes(1, n_reduce)
        + _field_bytes(1, n_mul)
        + _field_bytes(1, n_add)
        + _field_str(2, "DummyYoloGraph")
        + _field_bytes(5, zero_init)
        + _field_bytes(5, pred_init)
        + _field_bytes(11, images_vi)
        + _field_bytes(12, output_vi)
    )

    opset_msg = _field_str(1, "") + _field_varint(2, 13)
    model_msg = (
        _field_varint(1, 7)  # ir_version = 7
        + _field_str(2, "safety_engine_test_generator")
        + _field_bytes(7, graph_msg)
        + _field_bytes(8, opset_msg)
    )
    output_path.write_bytes(model_msg)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    output_path = repo_root / "models" / "dummy_yolo.onnx"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pred_bytes = _build_prediction_buffer()
    try:
        _create_with_onnx_package(output_path, pred_bytes)
        backend = "onnx.helper"
    except ImportError:
        _create_with_wire_serializer(output_path, pred_bytes)
        backend = "pure-Python ONNX protobuf serializer"

    print(f"Generated {output_path} ({output_path.stat().st_size} bytes) via {backend}.")


if __name__ == "__main__":
    main()
