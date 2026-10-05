#!/usr/bin/env python3
"""
Export RAFT model to ONNX for C++ runtime migration.

Usage:
  python3 export_raft_onnx.py --ckpt /path/to/raft-kitti.pth --output raft_kitti.onnx
"""

import argparse
import os
import sys

import torch
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
sys.path.append(os.path.join(SCRIPT_DIR, "core"))

from core.raft import RAFT  # type: ignore


class _Args:
    def __init__(self):
        self.small = False
        self.mixed_precision = False
        self.alternate_corr = False
        self.dropout = 0.0

    def __contains__(self, item):
        return hasattr(self, item)


class _RaftOnnxWrapper(nn.Module):
    def __init__(self, raft_model: nn.Module, iters: int):
        super().__init__()
        self.raft = raft_model
        self.iters = iters

    def forward(self, image1, image2):
        flow_low, flow_up = self.raft(image1, image2, iters=self.iters, test_mode=True)
        return flow_low, flow_up


def _align_to_multiple_of_8(v: int) -> int:
    return ((v + 7) // 8) * 8 # 对齐到8的倍数


def _normalize_state_dict(raw_state):
    if isinstance(raw_state, dict) and "state_dict" in raw_state:
        raw_state = raw_state["state_dict"]
    if not isinstance(raw_state, dict):
        raise RuntimeError("Checkpoint format is invalid: expected a state_dict-like mapping.")

    # Common case: checkpoints saved from DataParallel have "module." prefix.
    if any(k.startswith("module.") for k in raw_state.keys()):
        return {k[len("module."):]: v for k, v in raw_state.items()}
    return raw_state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="RAFT checkpoint .pth")
    parser.add_argument("--output", required=True, help="Output ONNX path")
    parser.add_argument("--height", type=int, default=376)
    parser.add_argument("--width", type=int, default=1240)
    parser.add_argument("--iters", type=int, default=12)
    args = parser.parse_args()

    model = RAFT(_Args())
    state = torch.load(args.ckpt, map_location="cpu")
    state = _normalize_state_dict(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        missing_preview = ", ".join(missing[:8]) if missing else "<none>"
        unexpected_preview = ", ".join(unexpected[:8]) if unexpected else "<none>"
        raise RuntimeError(
            "Checkpoint did not cleanly match RAFT model.\n"
            f"Missing keys ({len(missing)}): {missing_preview}\n"
            f"Unexpected keys ({len(unexpected)}): {unexpected_preview}"
        )
    print(f"[OK] Loaded RAFT checkpoint with {len(state)} tensors.")
    model.eval()
    wrapper = _RaftOnnxWrapper(model, args.iters).eval()

    export_h = _align_to_multiple_of_8(args.height) # 对齐到8的倍数
    export_w = _align_to_multiple_of_8(args.width) # 对齐到8的倍数  
    if export_h != args.height or export_w != args.width:
        print(
            f"[WARN] RAFT expects H/W multiples of 8. "
            f"Auto-adjust export shape from ({args.height},{args.width}) "
            f"to ({export_h},{export_w})."
        )

    img1 = torch.randn(1, 3, export_h, export_w, dtype=torch.float32) # 随机生成输入图像
    img2 = torch.randn(1, 3, export_h, export_w, dtype=torch.float32) # 随机生成输入图像

    with torch.no_grad():
        torch.onnx.export( # 导出ONNX模型
            wrapper, # 包装器
            (img1, img2),
            args.output, # 输出路径
            opset_version=17, # ONNX opset 版本
            input_names=["image1", "image2"], # 输入名称
            output_names=["flow_low", "flow_up"], # 输出名称    
            dynamic_axes={ # 动态轴
                "image1": {0: "batch", 2: "height", 3: "width"},
                "image2": {0: "batch", 2: "height", 3: "width"},
                "flow_low": {0: "batch", 2: "height", 3: "width"},
                "flow_up": {0: "batch", 2: "height", 3: "width"},
            },
        )

    print(f"[OK] Exported ONNX: {args.output}")
    print(f"[OK] Export shape: 1x3x{export_h}x{export_w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
