# METRICS BASELINE（统一指标口径）

用途：避免“每次测法不同导致结果不可比”，保证优化结论可复现、可用于面试与简历。

## 1) 固定测试环境

- 分支：
- commit：
- GPU：
- CUDA：
- TensorRT：
- 运行机器（本地/云端）：

## 2) 固定测试配置（必须一致）

- 数据范围：`--start` / `--count`
- 批大小：`--batch-size`
- 方法列表：`--methods`
- 环境变量（例如 variational/阈值相关）：

## 3) 固定命令（基线命令）

```bash
bash run_gpu_pipeline.sh --start 0 --count 50 --batch-size 10 --methods degraf_flow_interponet
```

```bash
bash run_gpu_pipeline.sh --start 0 --count 200 --batch-size 20 --methods degraf_flow_interponet
```

```bash
# 推荐：自动完成 variational=0/1 对照并写入本文件
python3 scripts/variational_ablation.py --start 0 --count 10 --methods degraf_flow_interponet --machine cloud
```

```bash
# 推荐：自动完成 off/full/gate 三组对照（门控阈值默认 3.0）
python3 scripts/residual_gate_ablation.py --start 0 --count 10 --methods degraf_flow_interponet --machine cloud --gate-thresh 3.0
```

## 4) 核心指标定义

- 速度指标：
  - 总耗时（s）
  - 单样本平均耗时（ms/frame）
  - 吞吐（frame/s）
- 精度指标（按项目实际产出填写）：
  - 指标名：
  - 指标值：

## 5) 结果记录表（改前 vs 改后）

| 日期 | commit | 改动点 | 配置摘要 | 改前速度 | 改后速度 | 改前精度 | 改后精度 | 结论 |
|---|---|---|---|---:|---:|---:|---:|---|
| 2026-02-24 | 646e005 | 基线对齐 | count=50,batch=10 | 待填 | 待填 | 待填 | 待填 | 待填 |

## 6) 判定规则（建议）

- 速度提升有效：在相同配置下重复 2 次以上，趋势一致。
- 精度可接受：不低于当前目标下限；若下降，需给出收益-风险说明。
- 若速度和精度冲突：优先满足“精度接近论文”，再在该前提下优化速度。

## 7) variational=0/1 对照实验记录

| 日期 | commit | 机器 | 配置 | EPE3d(v=0) | EPE3d(v=1) | AccS(v=0) | AccS(v=1) | AccR(v=0) | AccR(v=1) | Outlier(v=0) | Outlier(v=1) | Time(ms,v=0) | Time(ms,v=1) | 结论 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 待跑 | 待填 | cloud | `start=0,count=10,methods=degraf_flow_interponet` | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |

补充说明：
- 脚本会把原始日志保存到 `logs/ablation/`。
- 若你只想快速 smoke test，可先把 `--count` 改为 `2`，确认流程无误后再跑 `10` 或 `50`。

## 8) variational residual gate 对照实验

| 日期 | commit | 机器 | 配置 | EPE3d(off/full/gate) | Time(ms,off/full/gate) | AccS(off/full/gate) | AccR(off/full/gate) | Outlier(off/full/gate) | 结论 |
|---|---|---|---|---|---|---|---|---|---|
| 待跑 | 待填 | cloud | `start=0,count=10,methods=degraf_flow_interponet,gate_thresh=3.0` | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
