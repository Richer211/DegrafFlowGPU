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
        self.small = False # 是否使用小模型
        self.mixed_precision = False # 是否使用混合精度计算
        self.alternate_corr = False # 是否使用交替相关
        self.dropout = 0.0 # 丢弃率

    def __contains__(self, item): # 判断item是否在self中
        return hasattr(self, item) # hasattr判断item是否在self中


class _RaftOnnxWrapper(nn.Module):
    def __init__(self, raft_model: nn.Module, iters: int):
        super().__init__() # 初始化父类
        self.raft = raft_model # 这里的self.raft是RAFT模型, 调用RAFT模型的forward方法, 输入两张图像，输出光流
        self.iters = iters # 迭代次数

    def forward(self, image1, image2): # image1和image2是输入图像, 用于测试ONNX模型
        flow_low, flow_up = self.raft(image1, image2, iters=self.iters, test_mode=True) # 这里的self.raft是RAFT模型, 调用RAFT模型的forward方法, 输入两张图像，输出光流
        return flow_low, flow_up # 返回低分辨率光流和上采样光流


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
    parser = argparse.ArgumentParser() # 创建解析器
    parser.add_argument("--ckpt", required=True, help="RAFT checkpoint .pth") # 权重路径
    parser.add_argument("--output", required=True, help="Output ONNX path") # 输出路径  
    parser.add_argument("--height", type=int, default=376) # 高度
    parser.add_argument("--width", type=int, default=1240) # 宽度
    parser.add_argument("--iters", type=int, default=12) # 迭代次数
    args = parser.parse_args() # 解析命令行参数

    model = RAFT(_Args()) # 创建RAFT模型
    state = torch.load(args.ckpt, map_location="cpu") # 加载权重
    state = _normalize_state_dict(state) # 归一化权重
    missing, unexpected = model.load_state_dict(state, strict=False) # 加载权重
    if missing or unexpected: # 如果缺失或意外的权重, 则抛出异常
        missing_preview = ", ".join(missing[:8]) if missing else "<none>" # 缺失的权重
        unexpected_preview = ", ".join(unexpected[:8]) if unexpected else "<none>" # 意外的权重
        raise RuntimeError( # 抛出异常
            "Checkpoint did not cleanly match RAFT model.\n" # 权重不匹配
            f"Missing keys ({len(missing)}): {missing_preview}\n" # 缺失的权重
            f"Unexpected keys ({len(unexpected)}): {unexpected_preview}" # 意外的权重
        )
    print(f"[OK] Loaded RAFT checkpoint with {len(state)} tensors.") # 打印加载的权重数量
    model.eval() # 主要用于评估模型的性能，不进行训练
    wrapper = _RaftOnnxWrapper(model, args.iters).eval() # 创建包装器, 用于包装RAFT模型, 输入两张图像，输出光流.eval()表示评估模式

    export_h = _align_to_multiple_of_8(args.height) # 对齐到8的倍数,args.height表示高度
    export_w = _align_to_multiple_of_8(args.width) # 对齐到8的倍数,args.width表示宽度
    if export_h != args.height or export_w != args.width: # 如果高度或宽度不匹配, 则打印警告
        print( # 打印警告
            f"[WARN] RAFT expects H/W multiples of 8. " # 警告
            f"Auto-adjust export shape from ({args.height},{args.width}) " # 自动调整输出形状
            f"to ({export_h},{export_w})." # 输出形状
        )
    # 随机生成输入图像, 用于测试ONNX模型
    img1 = torch.randn(1, 3, export_h, export_w, dtype=torch.float32) # 随机生成输入图像 1表示批量大小, 3表示通道数, export_h表示高度, export_w表示宽度, dtype=torch.float32表示数据类型
    img2 = torch.randn(1, 3, export_h, export_w, dtype=torch.float32) # 随机生成输入图像

    with torch.no_grad():
        torch.onnx.export( # 导出ONNX模型
            wrapper, # 包装器
            (img1, img2),
            args.output, # 输出路径
            opset_version=17, # ONNX opset 版本
            input_names=["image1", "image2"], # 输入名称
            output_names=["flow_low", "flow_up"], # 输出名称    
            dynamic_axes={ # 动态轴 ，用于动态调整输入和输出的大小, flow_low表示低分辨率光流, flow_up表示上采样光流，作用是用于动态调整输入和输出的大小, 0表示批量大小, 2表示高度, 3表示宽度
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
