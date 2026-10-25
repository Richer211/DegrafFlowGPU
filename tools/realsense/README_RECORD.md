# RealSense 录制说明（v0）

## Mac 准备

```bash
# 建议用 conda/venv
pip install pyrealsense2 opencv-python numpy

# 确认是 USB3 且识别为 D435I（不要出现 USB2 降级名）
rs-enumerate-devices
```

固件若低于推荐版本，先用 Intel 工具升级。

## 录一段

```bash
cd tools/realsense
python3 record_realsense_seq.py \
  --out ~/realsense_seq \
  --name seq_box_slide_001 \
  --seconds 15 \
  --width 848 --height 480 --fps 30 \
  --preview
```

按 `q` 可提前结束。输出目录：

```text
~/realsense_seq/seq_box_slide_001/
  meta.json
  calib.json
  timestamps.csv
  color/000000.png ...
  depth/000000.png ...   # uint16，单位毫米，已对齐到彩色
```

## 拍摄怎么做

- 相机固定，不要手持晃。
- 桌子不用完全清空；背景尽量简单（木桌/纸即可），别太乱。
- **可以用手推物体**，第一版完全可以。
- 推荐片段：
  - A：静止 3 秒
  - B：单个盒子/瓶子，用手平稳推过画面（约 10 秒）
  - C：两个物体一起或依次推过

手会出现在画面里、也会有光流，第一版没关系；以后要更干净可以改成纸板拖动，把手挡在画面外。
