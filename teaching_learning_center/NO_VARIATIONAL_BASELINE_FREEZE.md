# No-Variational 阶段收口记录（冻结版）

日期：2026-03-11  
分支：`feature-optimize`

## 1) 当前可确认结论

- `feature`（Python/TCP）链路在 **no-variational** 条件下可复现到约 `EPE3d ~0.261x`。
- `feature-optimize`（C++/TRT）链路在 **no-variational** 条件下稳定在约 `EPE3d ~0.275x`（速度显著更快）。
- 两链路 dense 输出逐帧目录对比（200 帧）差异较大，说明不是简单的数值舍入误差。
- 当前剩余精度差距应视为“实现级流程不等价”问题，而非单个参数微调问题。

## 2) 已排除/已验证方向（证据摘要）

1. **TF checkpoint -> ONNX**  
   - 同输入低分辨率输出对比误差极小（`weighted_mean_epe` 约 `1e-6 ~ 1e-4` 量级）。
   - 结论：导出链本身不是主因。

2. **ONNXRuntime -> TRT lowres**  
   - 同输入 lowres 输出误差极小。
   - 结论：TRT 执行数值误差不是主因。

3. **InterpoNet 输入打包语义（image/mask）**  
   - Python vs C++ 输入对齐结果接近 0。
   - 结论：`image/mask` 打包非主因。

4. **InterpoNet edges downscale 语义**  
   - 已优化后 `edges` 输入差异显著降低（约 `0.007 -> 0.00003`），但 200 帧 EPE 没有等比例改善。
   - 结论：是次要因素，不是决定性主因。

5. **lowres->fullres 上采样 A/B（cubic/lanczos）**  
   - 200 帧指标变化很小（千分位级）。
   - 结论：非主因，已回退该实验开关代码。

6. **老工件回放（matches + edges.dat）到 C++/TRT**  
   - 200 帧仍在 `~0.275x`，未回到 `~0.261x`。
   - 结论：工件来源不是决定性主因。

## 3) 当前冻结“最佳可复现模式”（no-variational）

> 目标：先稳定复现，再进入下一阶段（variational 加速与收敛）

```bash
DEGRAF_ENABLE_VARIATIONAL=0 \
DEGRAF_RAFT_BACKEND=trt \
DEGRAF_INTERPONET_BACKEND=trt \
bash run_gpu_pipeline.sh --start 0 --count 200 --methods degraf_flow_interponet
```

## 4) 清理说明（本次）

- 已回退：`feat: add InterpoNet upsample A/B modes`（实验开关，收益不明显）。
- 保留：用于证据链复核的对比脚本与输入/输出对齐工具（后续可随时复用）。

## 5) 下一阶段（你后续有时间再做）

1. 在当前冻结 no-var 基线上，开启 variational 跑 200 帧，记录差距：  
   `EPE3d / Time` 与论文目标差值。
2. 仅围绕 variational 做性能优化（先时间，再看精度副作用）。
3. no-var 与 variational 两套命令和结果持续双轨记录，避免混淆。
