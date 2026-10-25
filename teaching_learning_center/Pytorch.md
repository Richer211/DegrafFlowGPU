# PyTorch（本项目课记）

配套：`external/RAFT/core/raft.py`、`external/RAFT/export_raft_onnx.py`  
更短的速查也可看：`PYTORCH_PROJECT_CHEATSHEET.md`

---

## 1. PyTorch 是什么？

**用 Python 写神经网络、用 CPU/GPU 做张量计算的框架。**

作用（对照本项目）：

| 作用 | 例子 |
|---|---|
| 定义网络 | `class RAFT(nn.Module)` |
| 前向计算 | `forward(image1, image2)` → 光流 |
| 训练（本仓库线上几乎不做） | loss → backward → 更新权重 |
| 存/读权重 | `.pth` + `load_state_dict` |
| 导出部署 | `torch.onnx.export` → `.onnx` |

框架价值：卷积、激活、混合精度、导出等已有实现，你主要写「数据怎么流过这些层」。

**和神经网络、深度学习的关系**

- 神经网络 = 模型结构本身  
- 深度学习 = 用深层网络 + 数据 + GPU 学习的技术路线  
- PyTorch = 写/跑这些网络的**工具**（工具 ≠ 模型本身）

---

## 2. 固定套路：建一个模型

```text
import torch
import torch.nn as nn

class RAFT(nn.Module):           # 继承 PyTorch 模型基类
    def __init__(self, args):    # 组装零件（此时没有照片）
        super().__init__()
        self.fnet = ...
        self.cnet = ...
        self.update_block = ...

    def forward(self, image1, image2, ...):  # 算法：输入→输出
        ...
        return ...
```

| 写法 | 意思 |
|---|---|
| `nn.Module` | 所有模型的基类；才能自动管参数、`train/eval`、`.to(device)` |
| `__init__` | 搭积木：注册层；**没有**图像输入 |
| `forward` | 真正计算；**你自己写**图怎么变成光流 |
| `model(x)` | 框架会调 `forward`（一般不用手写 `model.forward`） |

**面试常问**

- 为何继承 `nn.Module`？→ 参数管理、模式切换、设备迁移。  
- `__init__` vs `forward`？→ 组装 vs 计算。

**我曾搞混的点**

- `self.fnet = BasicEncoder(...)` 在 **`__init__`**，不在 `forward`；那一行没有两张照片。

---

## 3. 张量、形状、batch、算子

**张量**：有形状的多维数字块。图片常用 `N,C,H,W`。

| 维 | 名字 | 意思 |
|---|---|---|
| N | batch | 一次几份样本 |
| C | channel | 通道（RGB=3） |
| H | height | 高 |
| W | width | 宽 |

例：`torch.randn(1,3,H,W)` → 1 份、3 通道、高 H、宽 W。  
改成 `2,3,H,W` → 变的是 batch：一次两份一起算。

**为什么叫张量计算？**  
网络里的加减乘、卷积都是对这些数字块做运算；**不必把 128 维想成可看见的几何空间**，看形状即可。

**batch 和 GPU**  
N 更大 → 一次打包更多，GPU 更容易吃饱（也更吃显存）。直觉：算子按批量并行。

**算子（operator）**  
框架里的一步运算（卷积、ReLU、矩阵乘等）。作用：调用高度优化实现（常上 CUDA），不用手写 GPU 内核。

项目里常见张量操作：

| 写法 | 意思 |
|---|---|
| `.float()` | 转 float32 |
| `.contiguous()` | 内存连续 |
| `.detach()` | 断开梯度 |
| `torch.split(..., dim=1)` | 沿通道切开（`net` / `inp`） |

---

## 4. `eval`、层行为、`no_grad`、梯度

| 写法 | 管什么 |
|---|---|
| `model.train()` | 训练时的层规则 |
| `model.eval()` | **推理时的层规则**（不是「算准确率」这个动作本身） |
| `torch.no_grad()` | **不建计算图、不存梯度** |
| 默认训练前向 | 会建计算图，好让 `loss.backward()` 反传 |

**层行为是什么？**  
有的层在 train / eval 下做法不同：

- Dropout：train 随机丢；eval 关掉  
- BatchNorm：train 用当前 batch 统计；eval 用训练时攒下的稳定统计  

**`eval` 和 `no_grad` 不能互相替代**

- `eval` → 改层怎么算  
- `no_grad` → 关梯度  
导出/推理通常两个一起用。

**我曾问：`torch.grad()`？**  
日常更常见：默认有梯度，或 `torch.enable_grad()`；推理用 `no_grad`。一般不说 `torch.grad()` 这个入口。

---

## 5. 激活函数

给网络加**非线性**；没有它，很多层叠在一起仍像一个大线性变换。

| 本项目见过 | 作用直觉 |
|---|---|
| `tanh` | 压到大约 (-1,1) |
| `relu` | 负→0，正保留 |
| `softmax` | 收成「权重/概率」（如上采样 mask） |

常见位置：卷积/线性层**后面**。

---

## 6. 混合精度 `autocast` 与 FP16/FP32

```python
with autocast(enabled=self.args.mixed_precision):
    ...
```

| | FP32 | FP16 |
|---|---|---|
| 精度 | **更高、更稳** | 更粗 |
| 速度/显存 | 相对慢 | 通常更快更省 |

- `autocast`：**自动混合**——合适的用 FP16，不稳的仍 FP32（不是全程 FP16）。  
- 本项目只有 `mixed_precision=True` 才真正开；导出脚本默认关。  
- **≠** `raft_kitti_fp16.engine`：那是 TensorRT **编译**时定的 engine 精度，和源码里 `autocast` 不是同一开关。

---

## 7. 导出相关固定写法（`export_raft_onnx.py`）

| 写法 | 意思 |
|---|---|
| `model = RAFT(...)` | 按结构造空壳 |
| `torch.load(ckpt)` | 读 `.pth` |
| `load_state_dict` | 填权重 |
| `model.eval()` | 推理层行为 |
| `torch.no_grad()` | 不建图 |
| `torch.randn(1,3,H,W)` | 假输入，告诉导出器形状 |
| `torch.onnx.export(...)` | **导出 ONNX 的标准入口**（固定就这么用） |

`torch.onnx.export` 关键参数：模型、示例输入、输出路径、`input_names` / `output_names`，常加 `opset_version`、`dynamic_axes`。

只改 `raft.py` **不会**自动改 `.engine`：要重新 load 权重 → export ONNX → 再编 engine。

---

## 8. 产物对照

| 词 | 文件 |
|---|---|
| 结构 | `raft.py` |
| 权重 | `*.pth` |
| 中间格式 | `*.onnx` |
| GPU 编译产物 | `*.engine`（C++ TensorRT 读这个） |

跑光流时 `run_gpu_pipeline.sh` 设的是 `.engine` 路径，不是打开 `.py`。

---

## 9. 自测（答案见下）

1. `1x3x720x1280` 四个维？  
2. `eval` 和 `no_grad` 各管什么？  
3. `autocast` 是全 FP16 还是混合？和 fp16.engine 一回事吗？  
4. 改 `raft.py` 不重新导出，engine 变吗？

**答案**  
1. N=1, C=3, H=720, W=1280  
2. 层行为；梯度/计算图  
3. 混合；不是一回事  
4. 不变  

---

## 10. 建议学法

1. 先啃项目里出现的 API（本文）  
2. 再小练习：自己写两层卷积 `nn.Module`、假数据训练三行、`export` 一次  
3. 再接 TensorRT（见 `TensorRT.md`、`模型学习.md`）
