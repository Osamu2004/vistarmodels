# VIP

This adapter runs the official ICML 2026 VIP method (Visual-guided Prompt
Evolution) under the same dataset and metric contracts used by the other
Vistar open-vocabulary segmentation baselines.

VIP is training-free and has no VIP-specific task checkpoint. It uses a
frozen DINOv3 ViT-L/16 backbone, the public dino.txt vision head and text
encoder, and the public CLIP BPE vocabulary. The bootstrap pins the official
VIP source at `5bd25ee03ec25c1538622cf7da661e8c0461e769` and downloads the
three Meta assets into `/root/data/weight/vip`. DINOv3 weights are access
controlled by Meta. If the stable URLs reject an unauthenticated request,
request access through the official DINOv3 download page and either place the
approved files at the managed paths or pass the signed links as
`VIP_BACKBONE_URL` and `VIP_DINOTXT_URL`. The public BPE file remains fully
automatic.

The public VIP source leaves its local asset paths empty. `vip_model.py`
therefore constructs the same released network and injects explicit managed
paths without editing the third-party checkout. It preserves the released
DINOv3 ViT-L/16 backbone, dino.txt configuration, text templates, visual-guided
alias weighting, saliency-aware log-sum-exp aggregation, and overlapping
sliding-window inference.

## Setup

Use a dedicated environment. Install a CUDA-enabled PyTorch build compatible
with the server first; do not let `pip` replace a working RTX 50-series build
with VIP's historical CUDA 11.6 environment.

```bash
conda create -n vip python=3.10 -y
conda activate vip

# Install the CUDA-enabled torch/torchvision pair appropriate for the server.
python -m pip install -r requirements-vip.txt
bash scripts/bootstrap_vip.sh
python tools/check_vip_deps.py
```

No MMSeg or MMCV installation is required by this adapter. Dataset discovery,
native-extent metrics, visualization, and the IDGBR-compatible 3-pixel WFm
implementation are shared with the established SegEarth-OV evaluation path.

## Evaluation

Each entry supports `GPU_IDS`, `NPROC_PER_NODE`, `MAX_SAMPLES`, `SAVE_IMAGES`,
`STRICT_PROTOCOL`, `OVERWRITE`, and `OUTPUT_DIR`.

```bash
GPU_IDS=0 NPROC_PER_NODE=1 bash run_bash/vip_loveda.bash
GPU_IDS=0 NPROC_PER_NODE=1 bash run_bash/vip_flair.bash
GPU_IDS=0 NPROC_PER_NODE=1 bash run_bash/vip_uavid.bash
GPU_IDS=0 NPROC_PER_NODE=1 bash run_bash/vip_xbd_pre_building.bash
GPU_IDS=0 NPROC_PER_NODE=1 bash run_bash/vip_chn6_road.bash
```

Defaults use the public remote-sensing config geometry: keep-ratio resize to a
maximum side of 448, 336-pixel overlapping windows, and stride 112. The paper
appendix instead describes resizing the short side to 336 before 224/112
sliding inference, but the released VIP vision head hard-codes a 21-by-21
token grid and therefore requires 336-pixel windows. For reproducibility, the
adapter defaults to the executable public source and records the discrepancy
in `run_config.json`. `RESIZE_POLICY=paper_short_side INPUT_SIZE=336` exposes
the appendix resize rule while retaining the required 336-pixel window.

The five Vistar datasets are not the four remote-sensing datasets reported in
the VIP paper (iSAID, Vaihingen, Potsdam, and VDD). These runs are explicit
cross-dataset adaptations with fixed class-alias files and a universal
zero-threshold argmax. Dataset-specific tuning can be requested with
`PROBABILITY_THRESHOLD`, `TAU`, and `TEMPERATURE`, but tuned results must be
reported as such.

`LOW_CONFIDENCE_ACTION=auto` mirrors the released VIP post-processing rule.
Datasets with an explicit class-0 background (`LoveDA`, `UAVid`, `xBD-pre`,
and `CHN6-CUG`) map rejected pixels to class 0. FLAIR has no evaluated
background class, so rejected pixels are saved as ID 255 and retained as
false negatives in IoU, F1, accuracy, and pixel-accuracy denominators. This
avoids the incorrect historical behavior where nonzero thresholds changed
low-confidence FLAIR pixels into class-0 `building` predictions. The default
threshold remains zero, so the published default path is unchanged.

Every completed run writes `metrics.json`, `run_config.json`,
`predictions.jsonl`, `per_image_metrics.csv`, class-ID masks, RGB predictions,
RGB ground truth, and overlays. Primary table values are `miou` for LoveDA,
FLAIR, and UAVid; `building_iou` for xBD-pre; `road_iou` for CHN6-CUG; and
`wfm_3px_percent` for the boundary metric.
