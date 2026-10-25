# DeGraF 特征点

这一份只记特征点怎么算出来。稀疏配对见 `sparse_flow_overview.md`，空像素怎么填成稠密光流见 `flow.md`，ONNX / engine 见 `TensorRT.md`。

从「三个产物」那句接着记：特征点、稀疏光流、稠密光流是三样东西。不能收成一条永远不变的顺序「先特征点，再稀疏，再稠密」。RLOF 那条是先有第 10 帧的点，再跟踪成配对，再交给 EPIC。RAFT 那条是点和 RAFT 的稠密表都先有，在点上取样才得到稀疏配对，拿去算 EPE 的稠密光流是后来 InterpoNet 或 EPIC 重新填的。

## 图

点这一列就能打开。后两张的讲解在 `flow.md`，为了以后一个入口能找到，也放在这里。

- [一个窗口的三个产物](/Users/ganggang/.cursor/projects/Users-ganggang-Documents-DegrafFlowGPU/canvases/degraf-one-window.canvas.tsx)
- [亮度质心](/Users/ganggang/.cursor/projects/Users-ganggang-Documents-DegrafFlowGPU/canvases/degraf-centroid.canvas.tsx)
- [白圈、白点和蓝点](/Users/ganggang/.cursor/projects/Users-ganggang-Documents-DegrafFlowGPU/canvases/degraf-offset.canvas.tsx)
- [亚像素：蓝点再挪一截](/Users/ganggang/.cursor/projects/Users-ganggang-Documents-DegrafFlowGPU/canvases/degraf-subpixel.canvas.tsx)
- [RLOF 金字塔只改步长](/Users/ganggang/.cursor/projects/Users-ganggang-Documents-DegrafFlowGPU/canvases/rlof-pyramid.canvas.tsx)
- [RAFT 在点上取样](/Users/ganggang/.cursor/projects/Users-ganggang-Documents-DegrafFlowGPU/canvases/raft-lookup.canvas.tsx)
- [空格子的箭头来自旁边已知的箭头](/Users/ganggang/.cursor/projects/Users-ganggang-Documents-DegrafFlowGPU/canvases/sparse-to-dense.canvas.tsx)

## 卡住过的地方

窗口是不是边长 9 的一块？窗口默认 3×3，一共 9 个格子。步长默认 9，下一块从这里再隔 9 个像素，所以窗口彼此不重叠。一个窗口只留一个点。`run_gpu_pipeline.sh` 里 `WINDOW` 3、`STEP` 9。

三个数组是什么？`keypoint_x`、`keypoint_y`、`keypoint_response`。还不是稠密光流。

白点、白圈、蓝点各是什么？

- 白点是亮度质心：`Σ(坐标 × I) / Σ(I)`。亮的格子把质心拉过去。
- 白圈是窗口几何中心，和亮度无关。3×3 窗口若从像素 `(100, 80)` 起，中心是 `(101.5, 81.5)`。均匀亮度时质心落在格子坐标的正中，和这个几何中心会差半个像素，这是半像素定义，不是亮度造成的。
- `dx, dy = 2 × (质心 − 几何中心)`。蓝点 = 质心 + `(dx, dy)`，这一步还没加亚像素。
- 最终特征点 = 蓝点 + 亚像素。把白点、蓝点、亚像素再加一次会把质心算两遍。最终点不叫质心。

蓝点为什么能跑出 3×3？放大用了 2 倍。右列全亮的例子：质心 x 在窗口右缘，再加一份同样的偏移，蓝点就出了窗口。最终点也可以在窗口外面。亚像素上的 `× 0.5` 只把这一小项压在大约 1 像素内，不负责把最终点夹回窗口里。

亚像素那两个数怎么来的？格子的局部下标是 `0, 1, 2`，尺子放在 `1.5`。权重是每个格子的亮度除以窗口亮度和。偏移先累加 `权重 × (下标 − 1.5) × 0.5`，再除以权重和（通常是 1）。例子格子 `[[1,1,8]×3]`，亮度和 30，亚像素是 `(+0.10, −0.25)`。`+0.10` 是因为右列亮。三行一样时 y 仍然是负数，因为下标 `0, 1, 2` 相对 `1.5` 的加权并不对称，不是因为某一行更亮。蓝点 `(102.10, 80.00)` 加上它，特征点是 `(102.20, 79.75)`。

`keypoint_response` 是不是匹配有多准？`response = 幅度 × 窗口对比度`。幅度是白点到蓝点的距离，等于白圈到质心距离的两倍。对比度是窗口里最大亮度减最小亮度。越大表示窗口不太平、点离几何中心更远。后面的 RLOF / RAFT 不用这个分数做筛选。它不是测出来的匹配正确率。

对比度、幅度、边缘阈值（0.3、8、15）注释掉，是为了让点数和 CPU 一致吗？点数本来就会一样。CPU 在 step 8 里每个窗口都 `emplace_back` 一个点；坐标无效（例如 −1 或跑出图像）时，换成窗口中心，点还在。注释掉对比度那支判断，效果是低对比窗口保留算出来的坐标和 response，而不是先写成 −1 再被换成窗口中心。

显著性是灰度图吗？灰度只是中间结果。DoGoS 只在第 10 帧上做：金字塔缩小再从最小层放大回来，等于模糊；`absdiff(清晰, 模糊) / 和`，然后按开关减均值并归一化到 0–255。窗口里拿来当亮度 `I` 的，是这张 `saliency_mat`。结构亮、平坦暗。面试讲到「第 10 帧先变成显著性图，再进 `CudaDetectGradients`」即可。`ImagePyramid` 负责缩小放大。`ImageArray` 分配过，DoGoS 做差分时不用它。

README 八步要背吗？八步都在显著性图已经有了之后。面试一句带过 1–5 和 7：空图返回、矩阵大小等于窗口数、检测器再转一次灰度、拷到 GPU、用 float 是为了除法能出小数、三个数组拷回。展开讲的是第 6 步的质心、偏移、亚像素，以及第 8 步打包成 `KeyPoint` 和上面的无效坐标替换。

## 以后接着学时先看的代码

- `gpu/cuda/degraf_detector.cu`：`computeGradientsEnhancedKernel`，对比度提前返回是注释掉的；质心、`dx/dy`、亚像素、response 在同一次 kernel 里。
- `CudaDetectGradients`：无效坐标换成窗口中心，窗口仍然留一个点。
- `src/SaliencyDetector.cpp` 的 DoGoS：只处理第 10 帧。
