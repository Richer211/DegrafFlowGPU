# TensorRT：把 RAFT 和 InterpoNet 放进 C++

填密算法在 `flow.md`。这一份只记部署：文件从哪来、谁读、跑完拿到什么。

每帧的 `enqueueV3`、CHW、padding 到 8 的倍数，是被追问时的下一层。明天和以后开写之前，先守住下面这几件。

## 卡住过的地方

只说一句「放进 C++ 进程、在 GPU 上推理，起不来就退回 EPIC」够不够？这句是收束。接着能讲的是：为什么搬、离线变成哪两个文件、C++ 运行时做哪几步、做成之后少了什么进程。

为什么要这么做？DeGraF、配对、评测都在这个 C++ 程序里。RAFT 的权重在 PyTorch checkpoint，InterpoNet 的权重在 TensorFlow checkpoint。推理若留在原框架，每一帧都要另一套 Python 运行时，图像还要进出另一个进程。旧办法的名字还在：`RaftEngineTcpCompat.cpp`。那个文件现在标成停用，一进来就返回 false，不会去连 Docker，也不会编 engine。InterpoNet 有同名的一份，同样停用。现在走 `RaftEngineTRT.cpp` 和 `InterpoNetEngineTRT.cpp`。

核心是不是把「网络结构 + 训练好的权重」收成 ONNX，就可以直接在 GPU 上跑？ONNX 这一步对。还差一步才在 GPU 上跑：ONNX 要在目标显卡上编成 `.engine`。C++ 进程跑的是 `.engine`，不是 `.onnx`。网络外面的排版仍由 C++ 做。

`export_interponet_onnx.py` 里有什么？这个脚本只跑一次，写出一个 `.onnx`。四步写在文件头：

1. `model.getNetwork` 搭出原来的结构。三个入口是稀疏 `(du, dv)`、mask、边缘。高宽是原图按默认 8 倍缩小之后的尺寸。
2. `saver.restore` 把 checkpoint 填进去。三件套是 `.meta`、`.index`、`.data-00000-of-00001`。
3. `convert_variables_to_constants` 把权重冻成常量。冻完只剩前向，没有训练。
4. `tf2onnx` 写成 `.onnx`。

`export_raft_onnx.py` 同类：读 `.pth`，`torch.onnx.export`。两个输入是两帧图像，输出是光流。`args.output` 是命令行 `--output` 传来的磁盘路径。导出时图会经过 CPU 内存，留下的是那个路径上的文件，不是一份一直住在内存里的对象。

是 `RaftEngineTcpCompat.cpp` 还是 `RaftEngineTRT.cpp` 把 ONNX 编译成 `.engine`？两个都不是。编 engine 在 C++ 程序外面做一次，笔记里用的是 TensorRT 的 `trtexec`，而且必须在要跑的那张 GPU 上编。编完的文件是：

- `external/RAFT/models/raft_kitti_debug_fp32.engine`，没有它就用同目录的 `raft_kitti_fp16.engine`
- `external/InterpoNet/models/interponet_kitti_fp32.engine`

仓库里看不到，因为 `external/RAFT/.gitignore` 忽略了 `models/`，根目录 `.gitignore` 忽略了 `external/InterpoNet/models/`。权重和 engine 不跟源代码提交。

为什么 `.cpp` 里没有 `#include` 那个 engine？它不是源文件。运行时用路径字符串打开。RAFT 读环境变量 `DEGRAF_RAFT_ENGINE_PATH`，没设置时由 `run_gpu_pipeline.sh` 填成上面的路径。InterpoNet 读 `DEGRAF_INTERPONET_ENGINE_PATH`。`file.read(engine_data.data(), size)` 就是按这个路径从磁盘读进 CPU 这边的内存。文件不在，`init` 失败。读进来之后 `createInferRuntime`，再 `deserializeCudaEngine`，才得到可以执行的 context。

```333:336:src/inference/RaftEngineTRT.cpp
runtime_.reset(nvinfer1::createInferRuntime(logger_));
engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size()));
```

算子、合并层、显卡、显存分别是什么？

- 算子是网络里的一种运算，例如卷积。同一种运算在 GPU 上可以有几种实现，编 engine 时按这张卡选一种。
- 合并层：连续几个小运算若分开跑，中间结果要多写几次显存。能连着算完就合成一次。
- 显卡是机器上的 GPU 硬件。显存是这块卡上的存储。C++ 进程平时用的是 CPU 那一侧的内存。`cudaMemcpy` 在这两边搬这一帧的输入和输出。
- engine 和这张 GPU、这份 TensorRT 版本绑在一起。换架构差很多的卡，一般要重新编。

做成之后多了什么？评分主循环不再起 PyTorch / TensorFlow 进程。DeGraF 的 CUDA 和两次模型前向在同一个 C++ 进程里。当时对过 RAFT 取样得到的配对，C++ engine 和原来的 Python 差在千分之一量级，说明导出和输入摆放对齐了。那次对比说明搬运没有把模型弄歪，不是 RAFT 比 RLOF 准多少。没有产线 FPS 可以说。

编 engine 时的算子融合、fp16，可以说是 TensorRT 在编文件时做的。仓库里两条路都有：脚本优先找 RAFT 的 fp32 debug engine，否则用 `raft_kitti_fp16.engine`；InterpoNet 用的是 `interponet_kitti_fp32.engine`。网络结构是论文作者的，不是在这个仓库里从头训练的。

## 运行时每一帧，只记到这个粒度

C++ 先在 CPU 内存里把输入摆成网络要的形状，拷到显存，`enqueueV3` 跑完前向，再拷回 CPU。

- RAFT：两张照片进去，稠密光流出来。流水线接着只在 DeGraF 点上取样，得到稀疏配对。RAFT 失败时没有 EPIC 可退；只有 `DEGRAF_ALLOW_LK_FALLBACK` 打开才会改走 LK，脚本里默认是关的。
- InterpoNet：三张表进去（稀疏 `(du, dv)`、mask、边缘），低分辨率稠密光流出来，再放大。失败且允许回退时改走 EPIC。默认随后还有一步 variational，见 `flow.md`。

ONNX 里没有读图、读配对、读边缘、放大这些步骤。它们留在 C++ 里，必须和原来的 Python 预处理一致，否则网络本身再稳，交出去的光流也会偏。
