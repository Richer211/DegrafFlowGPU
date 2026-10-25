# 光流评估 EvaluateOptFlow

复习用。只保留这条评估门里问过、说过、改过的内容。场景流评估 `EvaluateSceneFlow` 只在和它对照时出现。

## 数据从哪进

图不是从 `degraf_detector.cu` 进来的。`.cu` 拿到的是已经在内存里的图。

入口是两个评估函数，它们是门：读图、按 `method` 名字呼叫下一步、把图画出去。特征点不在这两个函数里算。

- `EvaluateOptFlow::runEvaluation`：给稠密光流打分。
- `EvaluateSceneFlow::runEvaluation`：给场景流打分。

`degraf_flow_InterpoNet` 的返回值是一批稠密光流 `std::vector<cv::Mat>`。写到磁盘的是评测函数里的 `imwrite`。

一对图的字段在门里面临时声明。`i1`、`i2` 此时还是空的，`imread` 才把像素放进去。

```424:429:src/EvaluateOptFlow.cpp
struct ImagePairData {
    Mat i1, i2;
    String i1_path, i2_path, groundtruth_path;
    String num_str;
    int image_no;
};
```

```445:457:src/EvaluateOptFlow.cpp
data.i1_path = base_dir + "image_2/" + data.num_str + "_10.png";
data.i2_path = base_dir + "image_2/" + data.num_str + "_11.png";
// ...
data.i1 = imread(data.i1_path, 1);
data.i2 = imread(data.i2_path, 1);
```

`*_10.png` 是第 10 帧，`*_11.png` 是第 11 帧。`groundtruth_path` 是拿来打分的真值光流，不参与计算光流。读失败或两张图尺寸、通道不一致，这一对 `continue`，不进入后面的计算。

## 名字各自装什么

看过但当时读不懂的写法：

- `cv::` 是 OpenCV 这个名字下面的。
- `std::` 是 C++ 标准库这个名字下面的。
- `std::vector<T>` 是一串可以变长的数组，尖括号里是每个元素的类型。`std::vector<cv::Mat>` 是一串图，`std::vector<cv::Point2f>` 是一串点。
- `cv::Mat` 是一整张图。`data.i1` 是第 10 帧，`data.i2` 是第 11 帧。
- `cv::Point2f` 是一个点，只有小数的 `x` 和 `y`。
- `cv::KeyPoint` 是检测器交出的一个特征，里面有坐标，还有响应强度。
- `std::vector<cv::KeyPoint>` 是一串特征。
- `cv::KeyPoint::convert(keypoints, points)` 只把每个特征的 `x, y` 抄进 `points`。后面的匹配只要坐标。

```133:133:src/FeatureMatcher.cpp
cv::KeyPoint::convert(gpu_gradient_detector->GetKeypoints(), points);
```

图是 `Mat`。点是 `Point2f`。这两样从一开始就不是同一种东西。

| 阶段 | 装什么 | 类型 |
|---|---|---|
| 两帧原图 | 像素 | `cv::Mat` |
| DeGraF 刚检出 | 坐标加响应强度 | `std::vector<cv::KeyPoint>` |
| 交给匹配之前 | 只留坐标 | `std::vector<cv::Point2f>` |
| 稀疏光流 | 起点一组、终点一组 | `SparseFlowMatches`，或 `points_filtered` 配 `dst_points_filtered` |
| 稠密光流 | 每个像素一个二维位移 | `cv::Mat`，类型 `CV_32FC2` |
| 场景流 | 每个像素一个三维位移 | `cv::Mat`，类型 `CV_32FC3` |

## 分数有两套

光流分数在 `main` 里打出来的是平均 EPE、R2.0、R3.0、Time、STD。结构体里还有背景、前景和更多阈值，面试先用打印出来的这几个。

```28:42:include/EvaluateOptFlow.h
struct OptFlowMetrics
{
    float EPE;
    float Fl_bg;
    float Fl_fg;
    float Fl_all;
    float std_dev;
    float R05;
    float R1;
    float R2;
    float R3;
    float R5;
    float R10;
    double time_ms;
    int image_no;
};
```

```301:307:src/main.cpp
cout << "Average EPE: " << avg_EPE / count << "\n";
cout << "Average R2.0: " << avg_R2 / count << "\n";
cout << "Average R3.0: " << avg_R3 / count << "\n";
cout << "Average Time: " << avg_time / count << " ms\n";
cout << "Average STD: " << avg_std / count << "\n";
```

EPE 是端点误差：估出来的稠密光流 `flow`，和 `ground_truth` 上每个像素的位移比距离。越小越好。`flow` 就是估出来的那张 `CV_32FC2`。

用来算 EPE 的图写在这里：

```624:626:src/EvaluateOptFlow.cpp
Mat kittiFlow = convertToKittiFlow(flow);
String output_path = "../data/outputs/"+ method+ "/" + data.num_str + "_10.png";
imwrite(output_path, kittiFlow);
```

`flow.empty()` 的那一对会跳过，不打分。

另一处 `imwrite(data.i1)`、`imwrite(data.i2)` 写的是原始图像对，不是拿去算 EPE 的光流。旁边还有特征点图、稀疏箭头图，那是展示，不是分数本身。

场景流分数在 `SceneFlowMetrics`：`EPE3d`、`AccS`、`AccR`、`Outlier`、`time_ms`。场景流是稠密光流再加上视差之后的结果，单位和光流的像素误差不是同一套。

## 门按名字呼叫哪条流水线

两条路径最后都交出稠密光流。`EvaluateOptFlow` 给这张图打 EPE。`EvaluateSceneFlow` 可以吃这两条里任意一条产出的稠密光流。

不能说成「光流评估用 RLOF，场景流评估用 RAFT」。

- 一批多于一帧，并且 `method == "degraf_flow_interponet"`：走 DeGraF（CUDA）+ RAFT 查表 + InterpoNet。
- `degraf_flow_rlof`：走 DeGraF + RLOF + EPIC。
- 只算一帧时，名字虽然是 `degraf_flow_interponet`，这个函数里改走 RLOF。

```583:586:src/EvaluateOptFlow.cpp
else if (method == "degraf_flow_interponet") {
    // Single frame InterpoNet redirects to RLOF
    FeatureMatcher algo;
    algo.degraf_flow_RLOF(data.i1, data.i2, flow, 127, (0.05000000075F), true, (500.0F), (1.5F), data.num_str);
```

DeGraF 点是这条流水线里自己这一端的核心。RLOF、EPIC、RAFT、InterpoNet 是接进去的现成方法。点可以留着，跟踪和填密可以换，换完仍然用同一套 EPE 看有没有变好。

## 当时说错、后来改过的

1. 数据从 `.cu` 进来，因为要先做特征点。图是 `imread` 进来的。
2. 特征点是 `cv::Mat`。特征点先是 `KeyPoint`，抽出坐标后才是 `Point2f`。`Mat` 是整张图，也是后面的稠密光流。
3. 稀疏光流的类型是一个 `cv::Point2f`。稀疏这一步是两组点：起点一组，终点一组。
4. `computeGradientsEnhancedKernel` 产出稠密光流，结果从 `degraf_flow_InterpoNet` 写出磁盘。kernel 写出的是关键点。`degraf_flow_InterpoNet` 返回稠密光流给调用方。写出磁盘的是 `imwrite`。
5. `EvaluateOptFlow` 像主持人，自己决定并完成各段计算。它是门，算特征点的是后面的函数。
6. 光流用 RLOF+EPIC，场景流用 CUDA DeGraF+RAFT+InterpoNet。两条都产出稠密光流；场景流是在光流上再加视差。一帧的 InterpoNet 名字会改走 RLOF。
7. `imwrite(data.i1)` 和 `imwrite(data.i2)` 就是各阶段结果。那两行是原图。拿去算 EPE 的是 `../data/outputs/<method>/<编号>_10.png`。
