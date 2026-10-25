# Mac 实时 Live（传送带 + RealSense）

Mac 没有 NVIDIA GPU。实时状态在 Mac 上用 Farneback 轻量光流；RAFT / InterpoNet / TensorRT 仍在 AutoDL 上离线跑。两边共用同一套 IDLE / MOVING + MAIN 框。

已在俯拍、固定机位下验证：钥匙、钥匙扣、电池放上皮带后变为 MOVING 并跟上物体；掉进收集盒后回到 IDLE。空皮带常保持 IDLE（红外点阵相对相机不动，绿皮带本身纹理弱）。

## 架构

```
RealSense D435I（俯拍，三脚架固定）
    ↓
live_grab_rs          C++，sudo 取流，原子覆盖最新帧
    ↓
/tmp/degraf_live/{color.jpg, depth_mm.u16, calib.json, frame_id.txt}
    ↓
realtime_status.py --mode live
    Farneback（约 0.08s 一对帧，适配较快且不可调速的小皮带）
    → 场景流 / 光流位移
    → motion_alarm（8mm、最小块 1200px、亚像素噪声抑制、短时保持）
    ↓
状态窗：实时画面 | 点云场景流 | IDLE/MOVING 与位移数据
```

## 依赖

- 已编译的 librealsense（与 `record_png_seq` 相同路径）
- Python 3 虚拟环境：`opencv-python`、`numpy`
- 同目录：`motion_alarm.py`、`viz_scene_flow.py`

Mac 系统 Python 受 PEP 668 保护，不要往系统环境 `pip install`。

```bash
cd ~/librealsense/tools/realsense
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install opencv-python numpy
```

## 1. 编译取流

```bash
cd ~/librealsense/tools/realsense
c++ -O2 -std=c++14 live_grab_rs.cpp -o live_grab_rs \
  -I"$HOME/librealsense/include" \
  -I"$HOME/librealsense/third-party" \
  -I"$HOME/librealsense/third-party/stb" \
  -L"$HOME/librealsense/build/Release" \
  -lrealsense2 -Wl,-rpath,"$HOME/librealsense/build/Release"
```

## 2. 运行

先退出 RealSense Viewer / `rs-capture`。相机用 USB3。

```bash
# 终端 A
sudo ./live_grab_rs --out /tmp/degraf_live

# 终端 B
source .venv/bin/activate
python3 realtime_status.py --mode live --live-dir /tmp/degraf_live
```

启动行应类似：`pair_dt=0.08s  sf>=8.0mm  min_blob=1200  hold=5`。

`q` 退出状态窗，`Ctrl+C` 停取流。两者不要同时开着 Viewer。

## 3. 预期

| 情况 | 状态 |
|------|------|
| 静止，或只有空皮带在转 | IDLE |
| 钥匙 / 电池 / 钥匙扣在皮带上移动 | MOVING + MAIN 框 |
| 物体掉进盒子、画面静下来 | 约 0.4 秒内回到 IDLE |

## 常见问题

| 现象 | 处理 |
|------|------|
| `RS2_USB_STATUS_ACCESS` / No device | `sudo`；退出 Viewer；拔插后换 USB3 口 |
| warmup frames=0 | 拔插相机；程序会自动再试一次 |
| Python 一直等 calib | 先确认终端 A 已写出 `/tmp/degraf_live/calib.json` |
| `No module named cv2` | 使用上面的 `.venv`，不要用系统 Python |
| 静止桌面误报 MOVING | 不要把阈值降到 3mm 以下；那是桌面噪声 |
| 物体一闪一闪 | 保持默认 `pair_dt=0.08`。皮带偏快，间隔拉到 0.15s 光流会跟丢 |

## 和离线的分工

| 用途 | 在哪 |
|------|------|
| 实时状态 | Mac，本说明 |
| 无相机回放已有序列 | `realtime_status.py --mode replay` |
| RAFT / InterpoNet 高质量光流 | AutoDL，用 `record_png_seq` 的序列 |
