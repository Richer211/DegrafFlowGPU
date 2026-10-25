#!/usr/bin/env python3
"""Single entry point: industrial 4-panel demo JPGs only.

Keeps the product viz clean. Debug scripts (viz_flow_filtered / viz_scene_flow /
viz_scene_flow_xyz / export_pointcloud_sf) stay in this folder but are optional.

Example:
  python3 export_demo.py \\
    --seq-kitti /root/autodl-tmp/data/realsense_kitti/seq_box_slide_002 \\
    --flow-dir /root/autodl-tmp/DegrafFlowGPU/data/outputs/degraf_flow_interponet \\
    --out /root/autodl-tmp/DegrafFlowGPU/data/outputs/realsense_viz/seq_box_slide_002/demo
"""

from __future__ import annotations

import sys

from export_pointcloud_easy import main as easy_main


def main() -> int:
    # Ensure industrial defaults if caller omits thresh (easy default is already 0.02).
    return easy_main()


if __name__ == "__main__":
    # Re-exec through easy's argparse using the same argv (script name ignored).
    sys.argv[0] = "export_pointcloud_easy.py"
    raise SystemExit(main())
