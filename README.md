# GPU-Based Scene Flow with DeGraF, RAFT, and InterpoNet

This repository implements a **GPU-accelerated sparse-to-dense scene flow pipeline** built on **Dense Gradient-based Features (DeGraF)**.  
The pipeline combines:

- **CUDA DeGraF detector** for uniform feature extraction  
- **RAFT** for dense optical flow estimation  
- **InterpoNet** for learned edge-preserving interpolation  
- KITTI-compatible evaluation for both optical and scene flow  

This project extends the original **DeGraF-Flow** framework (Stephenson et al., ICIP 2019) with modern GPU and deep-learning modules.

---

## Project Structure

```
DEGRAF_FLOW_GPU/
├── include/                   # C++ headers
├── src/                       # Core CPU/C++ implementation
├── gpu/                       # CUDA modules (DeGraF detector, kernels)
├── external/                  # Third-party models (RAFT / InterpoNet)
│   ├── RAFT/                  # Cloned from RAFT GitHub
│   └── InterpoNet/            # Cloned from InterpoNet GitHub
├── data/
│   ├── data_scene_flow/       # KITTI 2015 dataset
│   ├── devkit_scene_flow/     # KITTI devkit evaluation tools
│   └── outputs/               # Flow predictions, visualizations, metrics
├── CMakeLists.txt             # C++/CUDA build config
└── README.md                  # Project documentation
```

---

## External Dependencies

### RAFT (ECCV 2020)
- Repository: [https://github.com/princeton-vl/RAFT](https://github.com/princeton-vl/RAFT)
- Kept for ONNX export only: `export_raft_onnx.py` and the network files under `core/` (`raft.py`, `update.py`, `extractor.py`, `corr.py`, `utils/utils.py`).
- Inference runs from a TensorRT `.engine` inside the C++ process. Training scripts, the demo, and the old TCP server are not in this branch.

### InterpoNet (CVPR 2017)
- Repository: [https://github.com/shayzweig/InterpoNet](https://github.com/shayzweig/InterpoNet)
- Kept for ONNX export: `export_interponet_onnx.py` and `model.py`.
- `SrcVariational/` is still compiled into the C++ binary for the optional variational refine.
- Inference runs from a TensorRT `.engine`. The original Python driver and TCP server are not in this branch.

---

##  Requirements

- **C++/CUDA**
  - CUDA ≥ 12.0  
  - OpenCV 4.9 (built with `optflow`, `ximgproc`)  
  - CMake ≥ 3.12, GCC ≥ 9
- **TensorRT** (optional, required for the RAFT / InterpoNet GPU path)
  - A prebuilt `.engine` for this GPU, or ONNX export plus `trtexec` on the target machine
  - Python is only needed again when re-exporting ONNX (PyTorch for RAFT, TensorFlow 1 for InterpoNet)

---

## Build (C++/CUDA Core)

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

This builds the CUDA-accelerated DeGraF detector and the C++ evaluation tools.

---

## Dataset Setup

Download the **KITTI 2015 Scene Flow dataset** and **devkit**:

- KITTI 2015: [http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php)

Place data as:

```
data/
├── data_scene_flow/
│   ├── training/
│   └── testing/
├── devkit_scene_flow/
└── outputs/
```

---

## Running the Pipeline

Point `DEGRAF_RAFT_ENGINE_PATH` and `DEGRAF_INTERPONET_ENGINE_PATH` at the engines built for this GPU, or use `run_gpu_pipeline.sh`, which sets those paths. Then:

```bash
./build/degraf_flow
```

This will:

- Extract DeGraF features (CUDA)
- Run RAFT in-process with TensorRT and sample at those features
- Densify the sparse matches with InterpoNet (EPIC if the engine fails to load)
- Write KITTI-format outputs under `data/outputs/`

---

## Evaluation

We use the official **KITTI devkit_scene_flow** for evaluation.

```bash
cd data/devkit_scene_flow
make
./evaluate_scene_flow ../outputs/ results.txt
```

Outputs include:

- Optical flow metrics: **Fl-bg, Fl-fg, Fl-all**
- Scene flow metrics: **EPE3D, AccS, AccR, Outlier %**
- Visualization: flow/error maps (PNG)

---

## References

- F. Stephenson, T. Breckon, I. Katramados,
  *DeGraF-Flow: Extending DeGraF Features for Accurate and Efficient Sparse-to-Dense Optical Flow Estimation*, ICIP 2019.
- Z. Teed, J. Deng,
  *RAFT: Recurrent All-Pairs Field Transforms for Optical Flow*, ECCV 2020.
- S. Zweig, L. Wolf,
  *InterpoNet: A Brain Inspired Neural Network for Optical Flow Dense Interpolation*, CVPR 2017.
- M. Menze, A. Geiger,
  *Object Scene Flow for Autonomous Vehicles*, CVPR 2015.

---

## License

Academic and research use only.

---

## Author

- Gang Wang
- Durham University, 2025

```mermaid
flowchart LR
    K["KITTI images + GT"]

    %% 推理 pipeline
    subgraph pipeline["Scene-flow pipeline (C++ & CUDA)"]
        D["CUDA DeGraF<br/>degraf_detector.cu"]
        R["RaftEngineTRT"]
        I["InterpoNetEngineTRT"]
        F["FeatureMatcher.cpp<br/>predicted optical flow"]
        S["SceneFlowReconstructor.cpp<br/>predicted scene flow"]
    end

    %% 评估模块
    subgraph eval["Evaluation"]
        EvalOF["EvaluateOptFlow.cpp"]
        EvalSF["EvaluateSceneFlow.cpp"]
    end

    main["main.cpp<br/>orchestrator"]

    %% data flow 只保留核心数据流
    K --> D
    D --> R
    R --> I
    I --> F
    F --> S

    %% evaluation 使用 KITTI GT + 预测结果
    F --> EvalOF
    S --> EvalSF
    K --> EvalOF
    K --> EvalSF

    %% main 只连到两个子系统，减少电线
    main --> pipeline
    main --> eval
 ```
