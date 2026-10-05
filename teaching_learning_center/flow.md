# 稀疏配对怎么变成稠密光流

特征点的算法在 `degraf_cu.md`。TensorRT 怎么把网络放进 C++ 在 `TensorRT.md`。更早的 RLOF / RAFT 取样记在 `sparse_flow_overview.md`。

图：

- [空格子的箭头来自旁边已知的箭头](/Users/ganggang/.cursor/projects/Users-ganggang-Documents-DegrafFlowGPU/canvases/sparse-to-dense.canvas.tsx)
- [RLOF 金字塔只改步长](/Users/ganggang/.cursor/projects/Users-ganggang-Documents-DegrafFlowGPU/canvases/rlof-pyramid.canvas.tsx)
- [RAFT 在点上取样](/Users/ganggang/.cursor/projects/Users-ganggang-Documents-DegrafFlowGPU/canvases/raft-lookup.canvas.tsx)

## 卡住过的地方

稠密光流好像和稀疏配对没关系？有关系。空像素自己没有起点和终点，位移是用已有配对估的。没有那些箭头，就没有东西可填。会觉得无关，是因为 RAFT 那条路上先出现过一张稠密表。那张表只用来在特征点上读出起点和终点。拿去算 EPE 的图，是 InterpoNet 或 EPIC 根据这些配对重新填出来的。配对一换，填出来的图就变。

图右边为什么有一排渐变箭头？那张图画的是 EPIC 这种折中，不是 InterpoNet 的算法。左格已知 `(du, dv) = (4, 0)`，右格已知 `(0, 2)`，正中间没有轮廓、离两边一样远时，示意写成 `(2, 1)`。真图很少这么整齐。

EPIC 是不是找到最近的一个点，照它抄？不是。`setK` 是留下多少个已经有箭头的点，近的权重大，这 K 个都参加。权重看的是距离，不是空格子和那个点长得像不像。亮度像不像，是前面 RLOF 配对时做的。

「中间有一条颜色分界」是什么？两帧照片里物体的交界，比如衣服轮廓。量距离时，过这种交界算更远。像素上只隔几格、但中间有轮廓的配对，权重变小。一件物体的位移因此不会直接涂到另一件上。

箭头存 `(x, y)` 还是 `(du, dv)`？存位移。稠密结果是和照片一样大的 `CV_32FC2`。格子自己的位置就是起点。已知配对写入时 `du = 终点.x - 起点.x`，`dv = 终点.y - 起点.y`。空格子写入的也是两个数，只是这两个数是折中出来的。需要终点再算 `(x + du, y + dv)`，终点不另存一份。图上「往右 4」就是 `(4, 0)`。

`setSigma` 是把照片做高斯模糊吗？它管权重随距离掉得有多快。照片模糊不是这一步。`setUsePostProcessing` 才是后面的平滑。

EPIC 是 K-means 吗？是 K 个最近邻按距离加权。k-means 是聚类，这里没有。OpenCV 内部叫 locally-weighted fitting，比「正中间正好一半一半」多一步拟合。面试说清「K 近邻、按距离加权、轮廓把过界的距离算远」就够。拟合公式在 OpenCV 里，仓库没重写。

调用就这一处。`k`、`sigma` 是函数参数，不是文件里写死的 100：

```645:652:src/FeatureMatcher.cpp
Ptr<ximgproc::EdgeAwareInterpolator> gd = ximgproc::createEdgeAwareInterpolator();
gd->setK(k);
gd->setSigma(sigma);
gd->setUsePostProcessing(use_post_proc);
gd->interpolate(prev, points_filtered, cur, dst_points_filtered, dense_flow);
```

InterpoNet 是不是把 EPIC 搬到 GPU？同一份工作：稀疏配对进，每个像素一个 `(du, dv)` 出。填法不同。EPIC 的规则写在 OpenCV 里，现场找这 K 个邻居。InterpoNet 是别人用 TensorFlow 训练好的网络，权重已经固定，一次把空格子填完。GPU 是这个前向跑在显卡上。引擎加载失败时，同一批配对才改交 EPIC。`DEGRAF_ALLOW_EPIC_FALLBACK` 默认开着。

「用大量例子训练过」是什么意思？训练时每次给一张大部分为空的 `(du, dv)` 图、一张 mask、一张边缘，旁边放这张图正确的稠密光流。卷积权重被调到填出来的结果接近那份答案。面试用的 engine 里权重不再改。它不会在现场重做 K 近邻平均，所以正中间那一格不会保证正好是 `(2, 1)`。卷积有几层不用背。

两帧照片是不是都送进网络？网络前向看的三张表都不是照片：

- 叫 `image` 的那张，两通道就是 `(du, dv)`。起点落在哪个格子，格子上写 `终点 - 起点`。空格子先写 0。
- `mask`：有配对是 `1`，空格子是 `-1`。
- `edges`：物体轮廓。优先读事先算好的 SED `edges.dat`；没有文件时用第 10 帧做 Canny。轮廓是单独一张图，让网络看见边界。

三张再按默认 8 倍缩小，拼成 4 个通道进网络。吐出的小图每个格子已经是 `(du, dv)`。C++ 用 `cv::resize`（`INTER_CUBIC`）放大到照片尺寸。第 11 帧的照片不进这三张表，两帧关系已经在 `(du, dv)` 里。写进格子的代码：

```640:642:src/inference/InterpoNetEngineTRT.cpp
full_u[idx] = d.x - s.x;
full_v[idx] = d.y - s.y;
full_valid[idx] = 1;
```

InterpoNet 放大之后就是最终结果吗？放大之后的那张是稠密光流。`InterpoNetEngineTRT.cpp` 里其余大段不是另一个网络，分成这几件：做上面三张表、加载 engine 并推理、放大、推理失败改 EPIC、默认再跑 variational、计时和调试输出。variational 在网络外面，用两帧照片把这张光流再修一次。开关 `DEGRAF_ENABLE_VARIATIONAL`，`run_gpu_pipeline.sh` 里默认是 1。被问到只需说到「默认还会再修一次」。变分怎么迭代还没学。

RAFT 也输出稠密光流，为什么还要 InterpoNet？RAFT 那张是完整的稠密表，流水线只用它在 DeGraF 点上取样，得到稀疏配对。交出去算 EPE 的，是 InterpoNet（或失败时的 EPIC）按这些配对填出来、再经默认 variational 的那张。

单帧名字叫 `degraf_flow_interponet` 时会改走 RLOF。一批多于一帧才走 DeGraF + RAFT 查表 + InterpoNet。
