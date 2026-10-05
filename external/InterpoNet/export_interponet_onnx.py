#!/usr/bin/env python3
"""
Export InterpoNet TF1 checkpoint to ONNX.
这个脚本只运行一次，产出一个 .onnx 文件。它按文件头的四步做：

1，用 model.getNetwork 把 InterpoNet 原来的网络结构搭出来，并留三个入口：稀疏 (du, dv)、mask、边缘。高宽是原图除掉默认的 8 倍缩小，例如 375×1242 变成 46×155。
2，用 saver.restore 把 TensorFlow checkpoint 里训练好的权重填进这张图。checkpoint 是三件套：.meta、.index、.data-00000-of-00001。
3，convert_variables_to_constants 把这些权重冻成常量。冻完之后，图里只剩「输入进来，算出稠密光流」，没有训练、没有再改权重。
4，tf2onnx 把这张冻住的图写成 .onnx。

Pipeline:
1) Build TF1 graph from model.getNetwork(...)
2) Restore variables from ckpt
3) Freeze graph (variables -> const)
4) Convert frozen graph to ONNX with tf2onnx
"""

import argparse
import os
import sys

import tensorflow as tf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

import model  # type: ignore  # noqa: E402


def _align_downscaled_size(height: int, width: int, downscale: int) -> tuple:
    if downscale <= 0:
        raise ValueError("downscale must be positive")
    h_trim = height - (height % downscale)
    w_trim = width - (width % downscale)
    if h_trim <= 0 or w_trim <= 0:
        raise ValueError("invalid downscaled shape; check height/width/downscale")
    return h_trim // downscale, w_trim // downscale


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_prefix", required=True, help="Path prefix of TF1 ckpt")
    parser.add_argument("--output", required=True, help="Output ONNX path")
    parser.add_argument("--height", type=int, default=375, help="Original image height")
    parser.add_argument("--width", type=int, default=1242, help="Original image width")
    parser.add_argument("--downscale", type=int, default=8, help="InterpoNet downscale factor") # 高宽是原图除掉默认的 8 倍缩小，例如 375×1242 变成 46×155。
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    args = parser.parse_args()

    meta_path = args.ckpt_prefix + ".meta"
    index_path = args.ckpt_prefix + ".index"
    data_path = args.ckpt_prefix + ".data-00000-of-00001"

    missing = [p for p in (meta_path, index_path, data_path) if not os.path.exists(p)]
    if missing:
        print("[ERROR] Missing TF1 checkpoint files:")
        for p in missing:
            print(f"  - {p}")
        return 1

    try:
        import tf2onnx  # type: ignore
    except Exception as e:
        print("[ERROR] tf2onnx is required but not available.")
        print("[HINT] Install with: python3 -m pip install tf2onnx")
        print(f"[DETAIL] {e}")
        return 2

    tf.compat.v1.disable_eager_execution()
    down_h, down_w = _align_downscaled_size(args.height, args.width, args.downscale)

    graph = tf.Graph()
    with graph.as_default():
        image_ph = tf.compat.v1.placeholder(
            tf.float32, shape=(None, down_h, down_w, 2), name="image_ph"
        )
        mask_ph = tf.compat.v1.placeholder(
            tf.float32, shape=(None, down_h, down_w, 1), name="mask_ph"
        )
        edges_ph = tf.compat.v1.placeholder(
            tf.float32, shape=(None, down_h, down_w, 1), name="edges_ph"
        )
        prediction = model.getNetwork(image_ph, mask_ph, edges_ph, reuse=False) # 获取网络预测结果
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=0) # 保存变量

    with tf.compat.v1.Session(graph=graph) as sess: # 创建会话
        saver.restore(sess, args.ckpt_prefix) # 用 saver.restore 把 TensorFlow checkpoint 里训练好的权重填进这张图。
        output_op_name = prediction.op.name # 获取输出操作名称
        frozen_graph = tf.compat.v1.graph_util.convert_variables_to_constants( # 把这些权重冻成常量。
            sess, graph.as_graph_def(), [output_op_name] # 冻结图
        )

    input_names = ["image_ph:0", "mask_ph:0", "edges_ph:0"] # 输入名称
    output_names = [f"{output_op_name}:0"] # 输出名称

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True) # 创建输出目录
    tf2onnx.convert.from_graph_def( # 把这张冻住的图写成 .onnx。
        frozen_graph, # 冻结图
        input_names=input_names, # 输入名称 image_ph:0, mask_ph:0, edges_ph:0
        output_names=output_names, # 输出名称 output_op_name:0  prediction:0
        opset=args.opset, # ONNX opset 版本
        output_path=args.output, # 输出路径
    )

    print("[OK] Exported InterpoNet ONNX:", args.output)
    print(f"[OK] Input shape: batchx{down_h}x{down_w}x(2/1/1), downscale={args.downscale}")
    print("[OK] Input tensors:", ", ".join(input_names))
    print("[OK] Output tensor:", output_names[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
