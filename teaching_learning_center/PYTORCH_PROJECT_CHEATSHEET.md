# PyTorch 速查（以 DegrafFlowGPU / RAFT 为先）

目标：先看懂**本项目用到的写法**，再做小练习补洞。  
不是完整 API 手册；面试够用的「固定套路」优先。

配套源码：

- `external/RAFT/core/raft.py` — 模型定义
- `external/RAFT/export_raft_onnx.py` — 导出 ONNX
- `external/RAFT/core/extractor.py` / `corr.py` / `update.py` — 子模块

---

## 0. PyTorch 是干什么的？

一句话：**用 Python 搭神经网络、用 GPU 算张量的框架。**

常见作用：

| 作用 | 本项目对应 |
|---|---|
| 定义网络结构 | `class RAFT(nn.Module)` |
| 前向计算（推理/训练） | `forward(image1, image2)` |
| 自动求梯度、训练更新 | 本仓库线上主流程几乎不训；原版 RAFT 会用 |
| 存/读权重 | `.pth` + `load_state_dict` |
| 导出部署中间格式 | `torch.onnx.export` → 再交给 TensorRT |

框架的价值：卷积、矩阵乘、激活、混合精度、导出等**不用从零手写 CUDA**；你主要写「数据怎么流过这些层」。

---

## 1. 项目里最固定的模型套路

```text
import torch
import torch.nn as nn

class RAFT(nn.Module):          # ① 继承模型基类
    def __init__(self, args):   # ② 组装零件（还没照片）
        super().__init__()
        self.fnet = ...
        self.cnet = ...
        self.update_block = ...

    def forward(self, image1, image2, ...):  # ③ 写算法：图→光流
        ...
        return flow_predictions
```

调用时写 `model(img1, img2)`，框架会进 `forward`（一般不必手写 `model.forward(...)`）。

**面试常问**

- 为什么要继承 `nn.Module`？  
  → 自动管理参数、`train/eval`、`.to(device)`。
- `__init__` 和 `forward` 区别？  
  → 前者搭积木，后者拿输入算输出。

---

## 2. RAFT `forward` 在干什么（纠正版流程）

不是每一层每一步都在「更新特征网和上下文网」。更准确是：

1. **预处理**：像素从 `0~255` 收到大约 `-1~1`；`contiguous()` 让内存排布连续。
2. **提特征一次**：`self.fnet([image1, image2])` → `fmap1, fmap2`（`extractor.py`）。
3. **建相关查找**：`CorrBlock(fmap1, fmap2)`（`corr.py`）——先建好，循环里再按坐标取。
4. **提上下文一次（只对 image1）**：`self.cnet(image1)` → 拆成 `net`（隐藏状态）和 `inp`（输入特征）。
5. **初始化坐标网格**：`coords0, coords1`；光流定义为 `flow = coords1 - coords0`。
6. **迭代更新（默认多次）**：每轮  
   - 用当前 `coords1` 查相关 `corr`  
   - `update_block` 产出 `delta_flow`（以及更新后的 `net`）  
   - `coords1 += delta_flow`  
   - 上采样得到更清晰的光流，放进列表  
7. **返回**：训练常返回每步预测列表；`test_mode=True` 时返回低分辨率流 + 上采样流。

**面试常问**

- RAFT 光流存在哪？  
  → 存在「两个坐标场的差」`coords1 - coords0`，不是一开始就有一张完整 dense flow 参数。

---

## 3. 本项目高频 API（先背这些）

### 3.1 张量与形状

| 写法 | 意思 | 项目例子 |
|---|---|---|
| `torch.randn(1,3,H,W)` | 随机张量，假输入 | 导出 ONNX 时造假图 |
| `N,C,H,W` | batch, 通道, 高, 宽 | 一对 RGB 帧常见 `1x3xHxW` |
| `tensor.float()` | 转成 float32 | `fmap1.float()` |
| `tensor.contiguous()` | 保证内存连续 | 导出/部分算子需要 |
| `tensor.detach()` | 断开梯度 | 迭代里 `coords1.detach()` |
| `torch.split(x, sizes, dim=1)` | 沿通道切开 | `net, inp = torch.split(cnet, ...)` |

### 3.2 模块与推理模式

| 写法 | 意思 |
|---|---|
| `model.eval()` | 推理模式（BatchNorm/Dropout 行为固定） |
| `model.train()` | 训练模式 |
| `model.load_state_dict(state)` | 把 `.pth` 权重装进结构 |
| `torch.load(path, map_location="cpu")` | 读 checkpoint |
| `with torch.no_grad():` | 不建计算图、不存梯度（推理/导出常用） |

**面试常问**：`eval()` 和 `no_grad()` 能互相替代吗？  
不能。`eval` 改层行为；`no_grad` 关梯度。导出/推理通常两个一起用。

### 3.3 激活与常用函数

| 写法 | 意思 |
|---|---|
| `torch.tanh` / `torch.relu` | 激活 |
| `torch.softmax(x, dim=...)` | 归一成概率分布（上采样 mask） |
| `F.unfold` | 把局部窗口展开放进通道（`upsample_flow`） |

这些来自 `torch` / `torch.nn.functional`（常 `import F`）。

### 3.4 混合精度 `autocast`

```python
from torch.cuda.amp import autocast
with autocast(enabled=self.args.mixed_precision):
    ...
```

| 点 | 说明 |
|---|---|
| 干什么 | **自动混合精度**：合适的算子用更低精度算，加快、省显存 |
| 常见精度 | 大量矩阵/卷积走 **FP16（半精度）**；不稳定的部分仍用 **FP32（单精度）** |
| 本项目 | 只有 `mixed_precision=True` 时才真正启用；导出脚本里默认 `False` |
| 注意 | 这是 **训练/PyTorch 推理加速** 手段；和最终 `.engine` 的 FP16/FP32 **不是同一个开关**（engine 在 TensorRT 编译时定） |

**面试常问**：FP16 一定比 FP32 好吗？  
更快更省显存，但可能数值更不稳；所以叫「混合」，不是全程砍成 FP16。

### 3.5 导出 ONNX

```python
torch.onnx.export(
    wrapper,
    (img1, img2),          # 示例输入，决定能追溯的计算图
    "raft.onnx",
    opset_version=17,
    input_names=["image1", "image2"],
    output_names=["flow_low", "flow_up"],
    dynamic_axes={...},    # 允许 batch/H/W 变化（仍有限制）
)
```

这是 PyTorch 导出 ONNX 的**标准入口**。  
之后 TensorRT 再把 ONNX 编成 `.engine`；C++ 读 engine，不再读 `raft.py`。

---

## 4. 建议学习顺序（同意你的计划）

**阶段 A（现在）——只学项目出现的**

1. `nn.Module` / `__init__` / `forward`  
2. `eval` + `no_grad` + `load_state_dict`  
3. `torch.onnx.export` 六个关键参数  
4. `autocast` 是什么、和 TensorRT FP16 的区别  
5. 对照 `raft.py` 把 forward 六步说顺

**阶段 B（有空做小练习）**

- 自己写一个超小 `nn.Module`（两层卷积）跑 `forward`  
- 假数据上写 3 行训练环：`backward` → `step` → `zero_grad`  
- 把小模型 `export` 成 onnx（不一定上 TensorRT）

**阶段 C（接 TensorRT 课）**

- `.pth` → `.onnx` → `.engine` → `deserializeCudaEngine`

---

## 5. 一张对照表：PyTorch 词 vs 你们流水线

| 你听到的词 | 文件/产物 |
|---|---|
| 模型结构 | `raft.py` |
| 权重 | `*.pth` |
| 中间部署格式 | `*.onnx` |
| GPU 编译产物 | `*.engine` |
| 线上跑光流 | C++ + TensorRT 读 engine |

---

## 6. 自测题（先不看答案）

1. `torch.randn(1,3,H,W)` 四个数字各代表什么？  
2. 为什么导出前通常要 `model.eval()` 且包在 `torch.no_grad()` 里？  
3. `autocast` 打开时，是「全部变成 FP16」还是「混合 FP16/FP32」？  
4. 改了 `raft.py` 但没重新 `export` + 编 engine，线上 `.engine` 会变吗？
