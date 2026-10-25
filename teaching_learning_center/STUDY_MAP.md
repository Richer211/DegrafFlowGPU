# STUDY MAP（算法/语言基础学习地图）

用途：每天从这里选 1 个主题学习 30-60 分钟，并回流到项目代码。

配套任务单目录：`teaching_learning_center/Daily_Study/`
每日文件：`DAILY_STUDY_TASK_YYYY-MM-DD.md`

## 学习原则（最快有效）

- 只学“今天能用到”的知识点，不做大而全浏览。
- 每次学习输出 3 件事：
  1) 一句话定义
  2) 一个最小代码例子
  3) 一个项目回流点（改哪里）
- 用 30-60 分钟短周期；卡住超过 20 分钟先记问题再继续。

## 优先级 A（当前最相关）

### A1. ONNX -> TensorRT 部署闭环（部署基础）
- 学什么：
  - ONNX 是前向推理图
  - TensorRT 是 NVIDIA 推理优化与执行引擎
  - ONNX 导出、engine 构建、C++ 加载三步
- 去哪里学：
  - ONNX 官方文档：`https://onnx.ai/`
  - TensorRT 官方文档：`https://docs.nvidia.com/deeplearning/tensorrt/`
  - 项目脚本：`external/RAFT/export_raft_onnx.py`
- 回流点：
  - `src/inference/RaftEngineTRT.cpp`

### A2. C++ 参数传递（语言基础）
- 学什么：
  - 按值传递 vs 引用传递 vs `const` 引用
  - 拷贝成本与可修改性
- 去哪里学：
  - cppreference（函数参数、引用）：`https://en.cppreference.com/`
  - LearnCpp（引用与 const 引用章节）：`https://www.learncpp.com/`
- 回流点：
  - `FeatureMatcher.cpp`
  - `RaftEngineTRT.cpp`

### A3. 复杂度与工程常数（算法思维）
- 学什么：
  - `O(n)` 背后的常数成本（内存分配、拷贝、sqrt）
  - 容器预分配 `reserve` 的意义
- 去哪里学：
  - MIT 6.006（复杂度基础，公开课）
  - CP-Algorithms（复杂度与数据结构）：`https://cp-algorithms.com/`
- 回流点：
  - 匹配点过滤循环与批处理向量管理

## 优先级 B（两周后补）

- OpenCV 数据布局与 `cv::Mat` 生命周期
- CUDA 基础：host/device、异步拷贝、流（stream）
- 推理精度：FP32/FP16/INT8 与精度-速度权衡

## 每日最小执行模板（复制到 DAILY_LOG）

- 今日主题：
- 学习来源：
- 30-60 分钟动作：
- 我理解的一句话：
- 项目回流点：
