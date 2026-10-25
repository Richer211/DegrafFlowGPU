#!/bin/bash
set -e

# Runs the 200-pair benchmark and regenerates table_i/table_ii only.
export DEGRAF_PROJECT_ROOT="${DEGRAF_PROJECT_ROOT:-/root/autodl-tmp/projects/DegrafFlowGPU}"
if [ ! -d "${DEGRAF_PROJECT_ROOT}" ]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  export DEGRAF_PROJECT_ROOT="${SCRIPT_DIR}"
fi
export DEGRAF_DATA_PATH="${DEGRAF_DATA_PATH:-/root/autodl-tmp/data/kitti/data_scene_flow}"
export DEGRAF_CALIB_PATH="${DEGRAF_CALIB_PATH:-/root/autodl-tmp/data/kitti/data_scene_flow_calib}"

cd "${DEGRAF_PROJECT_ROOT}"
bash run_gpu_pipeline.sh \
  --start 0 \
  --count 200 \
  --batch-size 20 \
  --methods degraf_flow_interponet,degraf_flow_rlof,DISflow_fast,deepflow

echo "[OK] Regression finished. Outputs:"
echo "  - data/outputs/table_i_optical_flow.csv"
echo "  - data/outputs/table_ii_scene_flow.csv"
