# DAILY STUDY TASK 2026-03-03

## 今日学习任务单（可勾选）

- [ ] 任务 1（语言基础）：`const` 引用与复制成本
  - 学习来源：
    - `https://en.cppreference.com/w/cpp/language/reference`
    - `https://www.learncpp.com/`
  - 预计时长：30 分钟
  - 完成标准：解释为什么 `const std::vector<cv::Mat>&` 更适合批处理输入，并举 1 个当前项目函数

- [ ] 任务 2（部署基础）：RAFT 的 ONNX 到 TensorRT 路径
  - 学习来源：
    - `external/RAFT/export_raft_onnx.py`
    - `https://docs.nvidia.com/deeplearning/tensorrt/`
  - 预计时长：30 分钟
  - 完成标准：写出你自己的两条命令（导出 ONNX、生成 engine），并说明输入输出文件路径

- [ ] 任务 3（项目回流）：RaftEngineTRT 后端切换与严格模式
  - 回流文件/函数：`src/inference/RaftEngineTRT.cpp::estimateMatchesBatch`
  - 最小动作（3-20行）：阅读并验证 `DEGRAF_RAFT_BACKEND / DEGRAF_RAFT_ENGINE_PATH / DEGRAF_ALLOW_LK_FALLBACK`
  - 完成标准：你能口述 3 个环境变量分别控制什么行为

## 今日输出（必须填写）

- 我今天学到的 3 个关键点：
  1.
  2.
  3.
- 我今天最卡的 1 个点：
- 明天继续的 1 个动作：
