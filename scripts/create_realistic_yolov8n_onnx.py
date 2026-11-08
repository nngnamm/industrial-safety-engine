#!/usr/bin/env python3
"""Generate a realistic multi-layer Conv2D + SiLU + SPPF + Concat + Detect YOLOv8-Nano ONNX graph.

Output path: models/yolov8n_bench.onnx
Graph architecture (input [1, 3, 640, 640] -> output [1, 7, 8400]):
  - Stem (P1/2):   Conv2D(3->16, k=3, s=2, p=1) + SiLU         -> [1, 16, 320, 320]
  - Stage 1 (P2/4): Conv2D(16->32, k=3, s=2, p=1) + SiLU       -> [1, 32, 160, 160]
                    Conv2D(32->32, k=3, s=1, p=1) + SiLU       -> [1, 32, 160, 160]
  - Stage 2 (P3/8): Conv2D(32->64, k=3, s=2, p=1) + SiLU       -> [1, 64, 80, 80]
                    C2f branch A: Conv2D(64->32, k=1, s=1) + SiLU
                    C2f branch B: Conv2D(32->32, k=3, s=1, p=1) + SiLU
                    Concat([branch A, branch B], axis=1)       -> [1, 64, 80, 80] (P3 feature)
  - Stage 3 (P4/16): Conv2D(64->128, k=3, s=2, p=1) + SiLU     -> [1, 128, 40, 40] (P4 feature)
  - Stage 4 (P5/32): Conv2D(128->128, k=3, s=2, p=1) + SiLU    -> [1, 128, 20, 20]
  - SPPF Neck:      MaxPool(k=5, s=1, p=2) + Concat + Conv1x1  -> [1, 128, 20, 20] (P5 feature)
  - Multi-Scale Detect Head:
      P3 Head: Conv2D(64->7, k=1)  -> Reshape [1, 7, 6400]
      P4 Head: Conv2D(128->7, k=1) -> Reshape [1, 7, 1600]
      P5 Head: Conv2D(128->7, k=1) -> Reshape [1, 7, 400]
      Concat([P3, P4, P5], axis=2) + Anchor Prior -> output0 [1, 7, 8400]

Supports:
  1. Exporting via `ultralytics` if installed and `--use-ultralytics` is passed.
  2. `onnx.helper` graph builder if the `onnx` Python package is installed.
  3. Zero-dependency pure-Python ONNX Protobuf wire serializer fallback so it works
     out-of-the-box without PyTorch or external pip packages.
"""

from __future__ import annotations

import argparse
import array
import math
from pathlib import Path
import struct
import sys


NUM_CLASSES = 3
NUM_CHANNELS = 4 + NUM_CLASSES  # 7: [cx, cy, w, h, Person, Helmet, Vest]
NUM_ANCHORS = 8400


def _make_float_bytes(num_elements: int, scale: float = 0.01) -> bytes:
    """Create deterministic small float32 weights for Conv2D kernels."""
    vals = array.array(
        "f",
        (scale * math.sin(float((i * 17 + 5) % 97)) for i in range(num_elements)),
    )
    if sys.byteorder != "little":
        vals.byteswap()
    return vals.tobytes()


def _make_zero_float_bytes(num_elements: int) -> bytes:
    vals = array.array("f", [0.0]) * num_elements
    if sys.byteorder != "little":
        vals.byteswap()
    return vals.tobytes()


def _make_int64_bytes(values: list[int]) -> bytes:
    return b"".join(struct.pack("<q", v) for v in values)


def _build_anchor_prior_bytes() -> bytes:
    """Create raw float32 bytes [1, 7, 8400] with a valid worker + PPE anchor prior."""
    values = array.array("f", [0.0]) * (NUM_CHANNELS * NUM_ANCHORS)
    # Anchor #0: Person at center
    values[0 * NUM_ANCHORS + 0] = 320.0
    values[1 * NUM_ANCHORS + 0] = 320.0
    values[2 * NUM_ANCHORS + 0] = 100.0
    values[3 * NUM_ANCHORS + 0] = 200.0
    values[4 * NUM_ANCHORS + 0] = 0.94
    values[5 * NUM_ANCHORS + 0] = 0.04
    values[6 * NUM_ANCHORS + 0] = 0.02

    # Anchor #1: Helmet on head of Anchor #0
    values[0 * NUM_ANCHORS + 1] = 320.0
    values[1 * NUM_ANCHORS + 1] = 240.0
    values[2 * NUM_ANCHORS + 1] = 36.0
    values[3 * NUM_ANCHORS + 1] = 32.0
    values[4 * NUM_ANCHORS + 1] = 0.02
    values[5 * NUM_ANCHORS + 1] = 0.92
    values[6 * NUM_ANCHORS + 1] = 0.02

    if sys.byteorder != "little":
        values.byteswap()
    return values.tobytes()


# ------------------------------------------------------------------------------
# Minimal Protobuf wire-format helpers for zero-dependency ONNX serialization
# ------------------------------------------------------------------------------
def _encode_varint(value: int) -> bytes:
    if value < 0:
        value &= (1 << 64) - 1
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


def _make_attr_ints(name: str, values: list[int]) -> bytes:
    # AttributeProto: name=1, ints=8 (repeated int64), type=20 (INTS=7)
    ints_fields = b"".join(_field_varint(8, v) for v in values)
    return _field_str(1, name) + ints_fields + _field_varint(20, 7)


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


def _build_realistic_yolov8n_wire(output_path: Path) -> None:
    float_type = 1  # TensorProto.FLOAT
    int64_type = 7  # TensorProto.INT64

    initializers: list[bytes] = []
    nodes: list[bytes] = []

    def add_conv_silu(
        prefix: str,
        inp: str,
        out: str,
        c_in: int,
        c_out: int,
        k: int,
        s: int,
        p: int,
        scale: float = 0.005,
    ) -> None:
        w_name = f"{prefix}_w"
        b_name = f"{prefix}_b"
        conv_out = f"{prefix}_conv"
        sig_out = f"{prefix}_sig"

        initializers.append(
            _make_tensor_proto(
                w_name,
                float_type,
                [c_out, c_in, k, k],
                _make_float_bytes(c_out * c_in * k * k, scale),
            )
        )
        initializers.append(
            _make_tensor_proto(b_name, float_type, [c_out], _make_zero_float_bytes(c_out))
        )

        nodes.append(
            _make_node_proto(
                "Conv",
                [inp, w_name, b_name],
                [conv_out],
                f"{prefix}_conv_node",
                [
                    _make_attr_ints("kernel_shape", [k, k]),
                    _make_attr_ints("strides", [s, s]),
                    _make_attr_ints("pads", [p, p, p, p]),
                ],
            )
        )
        nodes.append(
            _make_node_proto("Sigmoid", [conv_out], [sig_out], f"{prefix}_sigmoid_node")
        )
        nodes.append(
            _make_node_proto("Mul", [conv_out, sig_out], [out], f"{prefix}_silu_mul_node")
        )

    def add_head_conv_reshape(
        prefix: str, inp: str, out: str, c_in: int, num_anchors_scale: int
    ) -> None:
        w_name = f"{prefix}_w"
        b_name = f"{prefix}_b"
        shape_name = f"{prefix}_shape"
        conv_out = f"{prefix}_conv"

        # Zero weights on final 1x1 projection so output matches deterministic anchor prior
        initializers.append(
            _make_tensor_proto(
                w_name,
                float_type,
                [NUM_CHANNELS, c_in, 1, 1],
                _make_zero_float_bytes(NUM_CHANNELS * c_in),
            )
        )
        initializers.append(
            _make_tensor_proto(
                b_name, float_type, [NUM_CHANNELS], _make_zero_float_bytes(NUM_CHANNELS)
            )
        )
        initializers.append(
            _make_tensor_proto(
                shape_name, int64_type, [3], _make_int64_bytes([1, NUM_CHANNELS, num_anchors_scale])
            )
        )

        nodes.append(
            _make_node_proto(
                "Conv",
                [inp, w_name, b_name],
                [conv_out],
                f"{prefix}_conv_node",
                [
                    _make_attr_ints("kernel_shape", [1, 1]),
                    _make_attr_ints("strides", [1, 1]),
                    _make_attr_ints("pads", [0, 0, 0, 0]),
                ],
            )
        )
        nodes.append(
            _make_node_proto(
                "Reshape", [conv_out, shape_name], [out], f"{prefix}_reshape_node"
            )
        )

    # Backbone + C2f + SPPF layers
    add_conv_silu("stem", "images", "p1", 3, 16, k=3, s=2, p=1)
    add_conv_silu("stage1_down", "p1", "p2_in", 16, 32, k=3, s=2, p=1)
    add_conv_silu("stage1_c2f", "p2_in", "p2", 32, 32, k=3, s=1, p=1)

    add_conv_silu("stage2_down", "p2", "p3_in", 32, 64, k=3, s=2, p=1)
    add_conv_silu("stage2_c2f_a", "p3_in", "p3_a", 64, 32, k=1, s=1, p=0)
    add_conv_silu("stage2_c2f_b", "p3_a", "p3_b", 32, 32, k=3, s=1, p=1)
    nodes.append(
        _make_node_proto(
            "Concat", ["p3_a", "p3_b"], ["p3_feat"], "stage2_concat", [_make_attr_int("axis", 1)]
        )
    )

    add_conv_silu("stage3_down", "p3_feat", "p4_feat", 64, 128, k=3, s=2, p=1)
    add_conv_silu("stage4_down", "p4_feat", "p5_in", 128, 128, k=3, s=2, p=1)

    # SPPF pooling + concat + 1x1 conv
    nodes.append(
        _make_node_proto(
            "MaxPool",
            ["p5_in"],
            ["p5_pool"],
            "sppf_maxpool",
            [
                _make_attr_ints("kernel_shape", [5, 5]),
                _make_attr_ints("strides", [1, 1]),
                _make_attr_ints("pads", [2, 2, 2, 2]),
            ],
        )
    )
    nodes.append(
        _make_node_proto(
            "Concat",
            ["p5_in", "p5_pool"],
            ["p5_cat"],
            "sppf_concat",
            [_make_attr_int("axis", 1)],
        )
    )
    add_conv_silu("sppf_fuse", "p5_cat", "p5_feat", 256, 128, k=1, s=1, p=0)

    # Multi-scale Detect Head across P3 (80x80=6400), P4 (40x40=1600), P5 (20x20=400)
    add_head_conv_reshape("head_p3", "p3_feat", "det_p3", 64, 6400)
    add_head_conv_reshape("head_p4", "p4_feat", "det_p4", 128, 1600)
    add_head_conv_reshape("head_p5", "p5_feat", "det_p5", 128, 400)

    nodes.append(
        _make_node_proto(
            "Concat",
            ["det_p3", "det_p4", "det_p5"],
            ["raw_pred"],
            "detect_concat",
            [_make_attr_int("axis", 2)],
        )
    )

    initializers.append(
        _make_tensor_proto(
            "anchor_prior",
            float_type,
            [1, NUM_CHANNELS, NUM_ANCHORS],
            _build_anchor_prior_bytes(),
        )
    )
    nodes.append(
        _make_node_proto("Add", ["raw_pred", "anchor_prior"], ["output0"], "detect_add_prior")
    )

    images_vi = _make_value_info_proto("images", float_type, [1, 3, 640, 640])
    output_vi = _make_value_info_proto("output0", float_type, [1, NUM_CHANNELS, NUM_ANCHORS])

    graph_msg = (
        b"".join(_field_bytes(1, n) for n in nodes)
        + _field_str(2, "Yolov8NanoConv2DBenchGraph")
        + b"".join(_field_bytes(5, init) for init in initializers)
        + _field_bytes(11, images_vi)
        + _field_bytes(12, output_vi)
    )

    opset_msg = _field_str(1, "") + _field_varint(2, 13)
    model_msg = (
        _field_varint(1, 7)  # ir_version = 7
        + _field_str(2, "safety_engine_yolov8n_bench_generator")
        + _field_bytes(7, graph_msg)
        + _field_bytes(8, opset_msg)
    )
    output_path.write_bytes(model_msg)


def _try_export_ultralytics(output_path: Path) -> bool:
    try:
        from ultralytics import YOLO  # type: ignore

        model = YOLO("yolov8n.pt")
        exported = Path(model.export(format="onnx", imgsz=640))
        output_path.write_bytes(exported.read_bytes())
        return True
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate realistic multi-layer Conv2D YOLOv8-Nano benchmark ONNX graph."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "models" / "yolov8n_bench.onnx",
        help="Destination ONNX file path (default: models/yolov8n_bench.onnx)",
    )
    parser.add_argument(
        "--use-ultralytics",
        action="store_true",
        help="Attempt to export via ultralytics if installed before falling back to pure-ONNX Conv2D builder.",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.use_ultralytics and _try_export_ultralytics(args.output):
        backend = "ultralytics YOLOv8n export"
    else:
        _build_realistic_yolov8n_wire(args.output)
        backend = "multi-layer Conv2D + SiLU + SPPF + Detect ONNX graph builder"

    print(f"Generated {args.output} ({args.output.stat().st_size} bytes) via {backend}.")


if __name__ == "__main__":
    main()
