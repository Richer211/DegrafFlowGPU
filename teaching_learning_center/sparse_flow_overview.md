# 稀疏光流：从特征点到起点终点

复习用。从「第 11 帧不再跑 DeGraF」这段讲起。重点是问过、画过、改过的地方。

稀疏光流是留下来的配对。每个点有一个起点（第 10 帧上的特征点）和一个终点（它在第 11 帧上的位置）。所有这样的对放在一起，就是稀疏光流。`status` 丢掉的点不在名单里。整张图每个像素都有位移，那是后面填出来的稠密光流。

RLOF 和 RAFT 的差别只在终点怎么来。起点都是第 10 帧上 DeGraF 已经点过名的位置。

## RLOF：把已经点过名的点挪到第 11 帧

第 11 帧上不再跑一遍 DeGraF。RLOF 手里只有第 10 帧那些点，要回答的是「同一个点在第 11 帧里到了哪个像素」。

参数在调用 OpenCV 之前设好。算法本体不在这个仓库里。

```578:587:src/FeatureMatcher.cpp
rlof_param->useIlluminationModel = true;
rlof_param->useGlobalMotionPrior = true;
rlof_param->smallWinSize = 10;
rlof_param->largeWinSize = 11;
rlof_param->maxLevel = 4;
rlof_param->maxIteration = 30;
rlof_param->supportRegionType = cv::optflow::SR_FIXED;
cv::Ptr<cv::optflow::SparseRLOFOpticalFlow> proc =
    cv::optflow::SparseRLOFOpticalFlow::create(rlof_param);
```

### 窗口和 (du, dv)

假设 DeGraF 在第 10 帧给了 `(100, 80)`。

窗口是以这个点为中心的一小方格像素亮度。边长 11 来自 `largeWinSize = 11`：向左右上下各伸 5 个像素，大约盖住 `x = 95..105`、`y = 75..85`。单像素太容易重复，所以要比这一小块的明暗排列。

`(du, dv)` 是往右几格、往下几格。试 `(4, 1)`，就去第 11 帧的 `(104, 81)` 周围剪下同样大小的一块，和第 10 帧这块比亮度。差得小，这个位移就更像。留下差得最小的那次。终点是旧坐标加上位移：`(100+4, 80+1) = (104, 81)`。

`useGlobalMotionPrior = true`：旁边的点已经往右走了，这个点先从「往右」开始试。`SR_FIXED`：方框大小固定，不跟着纹理变大变小。

### 找的是亮度排列，像也不等于一定是同一个点

窗口里比的是每个像素有多亮，以及谁暗谁亮。

第 10 帧这一小块是 10、20、30，也就是暗、中、亮。第 11 帧某处是 20、30、40：每个数都大了 10，排列仍是暗、中、亮。开了光照模型，整块一起变亮算像。另一处是 30、10、20，数字再接近也是另一种排列，算不像。

`useIlluminationModel` 管的是两帧之间整块一起变亮或变暗。它不用 DeGraF 点上的响应强度，响应强度也不是这里说的光照。

算法只能保证「这里最像」，不能保证「这里一定是同一个点」。灯没大变、这一小块的样子也没变时，最像的位置通常就是它去的地方。纹路到处一样、被挡住、样子变了，另一个位置可能更像，点就会对错，或者被丢掉。所以后面要用真值算端点误差。

### 四层是四张图，不是只试四个位移

`maxLevel = 4` 是四档由粗到细的图。每一档里面还会反复改位移，最多 `maxIteration = 30` 次。30 是这一档自己的上限，不是四层一共四次。

第一档是把图缩小，不是把第 11 帧放大。画布为了看清，又把小图拉大了，所以格子显得很大。

「这一档 1 格 = 8 像素」的意思是：在缩小的图上挪 1 格，等于在原图上挪 8 像素。演示里的步长是 8、4、2、1。它们是步长，不是图案里有几块蓝色。

顺序是：

1. 最粗的小图上先找到大概位置，比如大约往右。小图上挪 1 格等于原图上挪一大截。
2. 下一档从上一档的结果开始，只在附近改，步长变成 4 像素。
3. 再后面是 2 像素、1 像素。

中间某一档已经盖住图案，后面的细档仍然会走，把位置收到 1 像素。停在中间档，位置只能准到那一档的步长。某一档里再挪也不更像，这一档可以提前停，然后仍然进入下一档。

缩小的图是搜索用的副本。第 11 帧一直都在。算法不把第 11 帧变成一张新图，结束时留下的是位移。

附近的亮度怎么对都不像，这个点 `status` 丢掉，没有终点。细档不会再到远处另找一块。更远的位置应该已经在粗档的大步子里看过。

蓝色 L 是画布上的记号，表示「同一块图案」，不是特征点，也不是新长出来的像素。最粗的 8 像素格子把同一个 L 挤进相邻两格，所以当时看见两块大蓝。格子收成 1 像素后，同一个 L 又散回很多小点。图案没有变多。

可重复点的图：`canvases/rlof-pyramid.canvas.tsx`。

## RAFT：先有整张位移表，再按点去读

第 11 帧上不滑动窗口。稠密光流图和照片一样大。照片每个格子存颜色，光流图每个格子存「这个格子里的东西往哪挪了」，也就是已经算好的 `du`（往右）和 `dv`（往下）。

这张图不是第 11 帧，里面没有特征点。它和第 10 帧的格子对齐。特征点坐标来自 DeGraF，例如 `(100, 80)`。读出 `du = 12`、`dv = 4` 之后，终点是加法：`(112, 84)`。

可点的图：`canvases/raft-lookup.canvas.tsx`。

### 格子缝上的四格，以及为什么不四舍五入

点正好在 `(100, 80)` 上时，仍然调用 `bilinearSample`。`wx = 0`、`wy = 0`，另外三格权重是 0，结果就是左上角那一格里写好的数。没有另一条更短的代码。

点在 `100.3, 80.2` 这种缝上，表里没有这个小数坐标。四格是左上、右上、左下、右下：

```183:208:src/inference/RaftEngineTRT.cpp
const int x0 = static_cast<int>(std::floor(x)); // 100
const int x1 = x0 + 1;                          // 101
const int y0 = static_cast<int>(std::floor(y)); // 80
const int y1 = y0 + 1;                          // 81
const float wx = x - static_cast<float>(x0);    // 0.3
const float wy = y - static_cast<float>(y0);    // 0.2
const float v00 = at(y0, x0); // 左上 (100, 80)
const float v01 = at(y0, x1); // 右上 (101, 80)
const float v10 = at(y1, x0); // 左下 (100, 81)
const float v11 = at(y1, x1); // 右下 (101, 81)
const float v0 = v00 * (1.0f - wx) + v01 * wx; // 上面这一横
const float v1 = v10 * (1.0f - wx) + v11 * wx; // 下面这一横
return v0 * (1.0f - wy) + v1 * wy;
```

没有 `v02`、`v03`。下标第一位是行，第二位是列。

`0.7` 就是 `1 - 0.3`。`100.3` 离 `100` 更近，所以上面这一横是 `0.7 * v00 + 0.3 * v01`。左格就是 `v00`。`v0` 是上面左右掺完。`v1` 是下面左右掺完。`wy = 0.2` 表示更靠近上面，所以返回值是 `v0 * 0.8 + v1 * 0.2`。这一个数才是读到的位移。

四格都是 `12` 时，掺完还是 `12`，终点看起来没变。四格不一样才会变。`(100, 80)` 的 `du` 是 `10`，`(101, 80)` 的 `du` 是 `20`，点在 `100.3`：四舍五入读到 `10`，掺完是 `13`。终点 x 一个是 `110.3`，一个是 `113.3`。

文件里 `v0`、`v1` 那两行公式重复写了一次。函数用一对，再加上 `return`。交出去的只有返回值。

### 为什么同一个点读两次

平面上的挪动要两个数。光流图每个位置有两层，不是两条硬件管道。第 0 层整张表放在前面，存 `du`。第 1 层接在后面，存 `dv`。

```953:954:src/inference/RaftEngineTRT.cpp
flow_chw[hw] = row[xx][0];
flow_chw[static_cast<size_t>(dense_flow.rows) * dense_flow.cols + hw] = row[xx][1];
```

```190:191:src/inference/RaftEngineTRT.cpp
const size_t idx = static_cast<size_t>(channel) * h * w + static_cast<size_t>(yy) * w + xx;
return chw[idx];
```

`channel = 0`，`idx` 落在前半张，读到 `du`。`channel = 1`，跳过前半张，读到同一个位置上的 `dv`。

特征点还是一个 `(x, y)`。传进去的是「去表的哪里读」，返回值才是位移。

```169:173:src/inference/RaftEngineTRT.cpp
float bilinearSample(const std::vector<float> &chw,
                     int channel,
                     int h, int w,
                     float y, float x,
                     bool zero_padding)
```

```990:997:src/inference/RaftEngineTRT.cpp
const float du = bilinearSample(flow_chw, 0, dense_flow.rows, dense_flow.cols, fy, fx, sample_zero_padding);
const float dv = bilinearSample(flow_chw, 1, dense_flow.rows, dense_flow.cols, fy, fx, sample_zero_padding);
const cv::Point2f d(
    p.x + (pad_aware_sampling ? du : (du * inv_sx)),
    p.y + (pad_aware_sampling ? dv : (dv * inv_sy)));
```

| 参数 | 这次传的 | 作用 |
|---|---|---|
| `chw` | `flow_chw` | 整张光流表 |
| `channel` | `0` 或 `1` | 读水平位移，或读垂直位移 |
| `h`, `w` | 光流图高和宽 | 找到那一格，并判断出界 |
| `y`, `x` | `fy`, `fx` | 特征点在表上的坐标。先传 y，再传 x |
| `zero_padding` | `sample_zero_padding` | 落到图外时，这一格按 0 算 |

## 两份名单就是稀疏光流

`push_back` 是往名单末尾再加一个点。两行一起执行，同一个位置编号配成一对。

RAFT：`p` 是起点，`d` 是查表加出来的终点。`src_points[0]` 配 `dst_points[0]`。`continue` 跳过的点不进名单。

```8:11:include/inference/IFlowInferenceEngine.h
struct SparseFlowMatches
{
    std::vector<cv::Point2f> src_points;
    std::vector<cv::Point2f> dst_points;
};
```

```999:1000:src/inference/RaftEngineTRT.cpp
matches.src_points.push_back(p);
matches.dst_points.push_back(d);
```

RLOF 变量名不同。`points[i]` 是起点，`currPoints[i]` 是找到的终点。只有 `status` 通过、位移小于 100 像素、两点都在图里，才加进去。

```614:621:src/FeatureMatcher.cpp
if (status[i] &&
    sqrt(...) < max_flow_length &&
    currPoints[i] 在第11帧内 && points[i] 在第10帧内)
{
    points_filtered.push_back(points[i]);
    dst_points_filtered.push_back(currPoints[i]);
}
```

## 为什么 KITTI 2015 上 RAFT 往往更稳

RLOF 只看大约 11×11 里的亮度排列。纹路重复时，另一块可能更像，终点就偏。

RAFT 先用两张整图给每个格子写位移，稀疏光流只是在 DeGraF 点上把这张表读出来。大位移、遮挡多时，小窗口「附近亮度还在」的假设经常不成立，所以 RAFT 通常更稳。位移小、遮挡少、这一小块又认得出来时，两者往往差不多。

表里的数也是估计出来的。确定的只是「查到的就是 RAFT 写进去的那个数」，不是传感器直接量出来的位移。

仓库笔记里有的数字，是 C++ TensorRT 读出的匹配和 Python RAFT 相差大约 `1e-3`，说明查表和 Python 一致，不是 RLOF 对 RAFT 的端点误差。你自己在 KITTI 2015 上看到 RAFT 更好，面试可以讲这个现象和上面的原因。没记在表里的百分比不要报。

## 当时说错、后来改过的

1. 第 11 帧也要用 DeGraF 再检一批点，RLOF 去配这些新点。第 11 帧不再检测。终点是旧点的新坐标。
2. 窗口是整个第 11 帧，或者光流图的一格也叫窗口。窗口只属于 RLOF 的 11×11。光流图的一格只存 `du, dv`。
3. `maxLevel = 4` 表示只能试 4 个 `(du, dv)`，四次之内必须找对。四是四张图。每一张里最多改 30 次。
4. 第一层是把第 11 帧放大，再把小方框放进去。第一档是把图缩小。方框步子大，是因为小图 1 格等于原图很多像素。
5. 画布上两块大蓝是新特征点。那是同一个 L 图案被粗格子挤进两格。格子变细后又散回小点。
6. 某一层找到了就可以停，四层里挑最像的一层。细层仍要走，结果用最细那一档的位移。各层不放在一起打分。
7. `status` 失败后再去别的地方找蓝框。细层只在粗层停下来的附近改。不像就丢掉。
8. 对上图案后还要把第 11 帧恢复成原图，恢复完才算结束。结束留下的是位移。第 11 帧没有被改掉。
9. 相似指的是 DeGraF 的响应强度，强度就是光照开关。比的是窗口里的亮度排列。光照开关只允许整块一起变亮或变暗。
10. 稠密光流图是第 11 帧，格子像窗口，要到第 11 帧的格子里找特征点。光流图对齐的是第 10 帧。终点用加法得到。
11. 周围四格掺完，终点还是 `(112, 84)`，所以和四舍五入一样。那次四格数字相同才会不变。数字不同时，四舍五入和掺的结果不一样。
12. `bilinearSample` 的输入是位移本身，一次算出 `(du, dv)`。输入是去表上哪个 `(y, x)` 读。`du` 和 `dv` 要各读一次。
13. `channel` 是两条硬件管道。它是同一张表的前后两层。
