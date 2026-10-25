# Mac arm64 录制（不依赖 pyrealsense2）

Apple Silicon 上 `pip install pyrealsense2` 通常不可用。  
你已编译好 `librealsense`，用 `rs-record` + `rs-convert` 即可。

## 0. 准备

- 相机接 **USB3**（`rs-enumerate-devices` 应显示 `D435I`，不要是 `USB2`）
- 相机固定，桌上放盒子/瓶子，准备用手推

```bash
export RS_BIN=~/librealsense/build/Release
"$RS_BIN/rs-enumerate-devices" | head -30
```

## 1. 录序列（推荐：`record_png_seq`，连续存 color/depth）

**不要用 `rs-save-to-disk`**：它只各存 1 张 PNG 就退出，不能当视频序列。

先把本目录的 `record_png_seq.cpp` 拷到 Mac（或从 AutoDL 的
`DegrafFlowGPU/tools/realsense/` 同步过来），再编译：

```bash
# 确认 stb 头文件位置（二选一能找到即可）
ls ~/librealsense/third-party/stb_image_write.h \
   ~/librealsense/third-party/stb/stb_image_write.h 2>/dev/null

cd /path/to/DegrafFlowGPU/tools/realsense   # 或你放 cpp 的目录

# 若头文件在 third-party/ 根下：
c++ -O2 -std=c++14 record_png_seq.cpp -o record_png_seq \
  -I"$HOME/librealsense/include" \
  -I"$HOME/librealsense/third-party" \
  -L"$HOME/librealsense/build/Release" \
  -lrealsense2 -Wl,-rpath,"$HOME/librealsense/build/Release" -lz

# 录制前：关掉 rs-capture / Viewer；USB 异常时拔插相机
sudo ./record_png_seq --out ~/realsense_seq --name seq_static_002 --seconds 5
sudo ./record_png_seq --out ~/realsense_seq --name seq_box_slide_002 --seconds 15
sudo ./record_png_seq --out ~/realsense_seq --name seq_two_boxes_002 --seconds 15

# 合格：color 与 depth 张数接近且都 > 50
ls ~/realsense_seq/seq_box_slide_002/color | wc -l
ls ~/realsense_seq/seq_box_slide_002/depth | wc -l
```

输出已是 v0 目录：`color/` `depth/` `meta.json` `calib.json` `timestamps.csv`。

**预览（不存盘）**：`sudo ~/librealsense/build/Release/rs-capture`  
Depth/Color 都约 30fps 即可。录制前务必关掉。

**备选 bag（不推荐作主路径）**：`rs-record` + `rs-convert`；此前出现过 Depth 几乎没导出。
## 2. bag 转图片

```bash
SEQ=~/realsense_seq/seq_box_slide_001
mkdir -p "$SEQ/raw"
"$RS_BIN/rs-convert" -i "$SEQ.bag" -p "$SEQ/raw/"
# 看导出了什么
ls "$SEQ/raw" | head -40
```

把 `ls` 前几十行发给我，我再给你「整理成 color/ depth/ meta.json / calib.json」的一步脚本。

## 3. 拍摄内容（提醒）

| 片段 | 做什么 |
|---|---|
| 静止 3 秒 | 可选 |
| 单物体用手推过 | 主测试 |
| 两物体推过 | 可选第二段，另存 `seq_two_boxes_001.bag` |

桌子不用清空；用手推可以。

## 4. 暂时不要做的事

- 不要再 `pip install pyrealsense2`（arm64 基本没有轮子）
- 不要对 `wrappers/python` 直接 `pip install .`
- 采数据优先；Python 绑定以后要用再 CMake `-DBUILD_PYTHON_BINDINGS=ON` 重编
