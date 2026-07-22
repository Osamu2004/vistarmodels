# Standalone SegEarth-OV evaluation

This adapter runs the official training-free SegEarth-OV model independently
of DynamicEarth.  It covers the four single-image benchmarks used by the
Vistar paper:

- LoveDA semantic segmentation;
- FLAIR #1 semantic segmentation;
- xBD pre-disaster building extraction;
- CHN6-CUG road extraction.

The model source is pinned to official SegEarth-OV revision
`3e22a969b32c6d751bdbba64a88a0b670e630f55`.  SimFeatUp is pinned to
`78a0ba70b1d6ea7283684a88c98ce338af4593ca`.  The default model is the
official CLIP ViT-B/16 + `jbu_one` SimFeatUp configuration with residual
removal, logit scale 50, overlapping 224-pixel windows, and stride 112.
The evaluator rejects a different or metadata-free SegEarth-OV checkout by
default; `--allow_unpinned_source` exists only for explicitly labeled debug
runs.

## Installation

Use a dedicated CUDA environment.  Install a CUDA-enabled PyTorch build that
matches the server first, then run:

```bash
cd /root/code/vistarmodels
python -m pip install -r requirements-segearth-ov.txt
bash scripts/bootstrap_segearth_ov.sh
python tools/check_segearth_ov_deps.py
```

The bootstrap clones and pins both official repositories, verifies the
bundled 5.69 MB `xclip_jbu_one_million_aid.ckpt`, downloads and verifies the
official OpenAI CLIP ViT-B/16 checkpoint, and builds SimFeatUp's CUDA and C++
adaptive-convolution extensions.  Set `MAX_JOBS` if extension compilation
needs a lower parallelism limit.

The managed defaults are:

```text
third_party/SegEarth-OV
third_party/SimFeatUp
/root/data/weight/segearth_ov/pretrained/ViT-B-16.pt
third_party/SegEarth-OV/simfeatup_dev/weights/xclip_jbu_one_million_aid.ckpt
```

Every path can be overridden through the variables shown in
`scripts/bootstrap_segearth_ov.sh` and `run_bash/segearth_ov_eval.bash`.

## Shared inference contract

The official MMSeg preprocessing is reproduced without forcing the datasets
into a second copy:

1. Read an RGB image at native extent. FLAIR uses bands 1--3 of its official
   five-band GeoTIFF.
2. Resize the image bilinearly with `keep_ratio=True` so its longest side is
   `INPUT_SIZE=448` by default.
3. Normalize with the official CLIP mean/std, run SegEarth-OV's overlapping
   `SLIDE_CROP=224`, `SLIDE_STRIDE=112` inference, and restore logits to the
   original image extent before argmax.
4. Accumulate the confusion matrix only at original resolution. No ground
   truth mask is resized for metric computation.

The class-ID prediction under `pred_mask/` is always saved as the restart
cache, even with `SAVE_IMAGES=0`.  `OVERWRITE=0` resumes matching cached
predictions; `run_config.json` prevents reuse when any inference-critical
setting differs.  `SAVE_IMAGES=1` additionally writes input PNGs, colorized
prediction/GT masks, GT class IDs, and overlays.

All four evaluators report IoU/mIoU, mF1, mAcc, pixel accuracy, per-class
metrics, and the shared IDGBR-compatible 3-pixel boundary WFm.  Aggregate
artifacts are `metrics.json`, `predictions.jsonl`, and
`per_image_metrics.csv`.

`FEATURE_UP=0` is only a dependency/debug fallback. It is not the official
SegEarth-OV configuration and must not supply paper results.

## LoveDA

- Population: all 1,669 official Val images from Urban and Rural.
- Layout: `LoveDA/Val/{Urban,Rural}/{images_png,masks_png}`. The flattened
  MMSeg `img_dir/val` + `ann_dir/val` layout is also accepted.
- GT: raw 0 is ignored; raw IDs 1--7 become evaluation IDs 0--6.
- Text groups: the official `background`, `building,roof,house`, `road`,
  `water`, `barren`, `forest`, `agricultural` vocabulary.
- Official model settings: probability threshold 0.3 and CLS-token lambda
  -0.3.

```bash
MAX_SAMPLES=2 DATA_ROOT=/root/data/LoveDA \
  bash run_bash/segearth_ov_loveda.bash
DATA_ROOT=/root/data/LoveDA \
  bash run_bash/segearth_ov_loveda.bash
```

## FLAIR #1

- Population: the complete 15,700-patch `flair#1-test` split, with the same
  ten-domain and 193-zone integrity checks used by the GSNet and RSKT-Seg
  adapters.
- Image: native 512x512 five-band GeoTIFF; only RGB bands 1--3 are passed to
  SegEarth-OV.
- GT: raw IDs 1--12 become evaluation IDs 0--11; raw 0, 13--19, and 255 are
  ignored.
- Vocabulary: the existing GSNet/RSKT-Seg 12-class order, including the exact
  model prompts `pervious-surface` and `impervious-surface`; result files use
  their human-readable forms with spaces.
- Model settings: probability threshold 0.0 and CLS-token lambda -0.3.

The original SegEarth-OV repository does not include a FLAIR configuration.
This is therefore a repository-local adaptation to the already established
12-class FLAIR comparison protocol.  Do not combine its newly computed WFm
with the published/reported 13.09 mIoU as if both came from one run; preserve
the new `metrics.json` as the source for every metric reported from this
adapter.

```bash
MAX_SAMPLES=2 DATA_ROOT='/root/data/FLAIR-1-2/data/flair#1-test' \
  bash run_bash/segearth_ov_flair.bash
DATA_ROOT='/root/data/FLAIR-1-2/data/flair#1-test' \
  bash run_bash/segearth_ov_flair.bash
```

## xBD-pre

- Population: 933 `*_pre_disaster` images from the official xBD test split.
- Raw layout: `test/images` plus `test/labels` JSON is accepted directly.
  The official SegEarth-converted `images_pre` + `targets_cvt_pre` layout is
  also accepted.
- JSON labels: `features.xy` WKT exteriors are rounded and filled with
  OpenCV, exactly reusing the RSKT-Seg/GSNet xBD helper.
- Classes: `background, building`; the primary paper metric is foreground
  `building_iou`.
- Official model settings: probability threshold 0.0 and CLS-token lambda 0.

The 448-pixel protocol is the default.  To run the separately reported
higher-resolution setting, use `INPUT_SIZE=896`; this creates a distinct
default output directory.

```bash
MAX_SAMPLES=2 DATA_ROOT=/root/data/xview2/test \
  bash run_bash/segearth_ov_xbd_pre_building.bash
DATA_ROOT=/root/data/xview2/test \
  bash run_bash/segearth_ov_xbd_pre_building.bash

# Optional higher-resolution protocol
INPUT_SIZE=896 DATA_ROOT=/root/data/xview2/test \
  bash run_bash/segearth_ov_xbd_pre_building.bash
```

## CHN6-CUG

- Population: all 903 validation images.
- Both the raw `images` + `gt` layout and the official SegEarth-OV
  `image_cvt` + `label_cvt` layout are accepted.
- GT: zero is background and every nonzero value is road.
- Classes: `background, road`; the primary paper metric is foreground
  `road_iou`.
- Official model settings: probability threshold 0.8 and CLS-token lambda
  -0.3.

```bash
MAX_SAMPLES=2 DATA_ROOT=/root/data/CHN6-CUG/val \
  bash run_bash/segearth_ov_chn6_road.bash
DATA_ROOT=/root/data/CHN6-CUG/val \
  bash run_bash/segearth_ov_chn6_road.bash
```

## Multi-GPU and dry runs

Each process owns one independent model and receives a deterministic strided
subset of files. Gloo synchronizes only result-file boundaries.

```bash
GPU_IDS=0,1 NPROC_PER_NODE=2 MAX_SAMPLES=2 \
  bash run_bash/segearth_ov_loveda.bash

DRY_RUN=1 BOOTSTRAP_SEGEARTH_OV=0 CHECK_DEPS=0 \
  bash run_bash/segearth_ov_flair.bash
```

Run a two-sample smoke test and inspect `pred_mask`, `gt_rgb`, and `overlay`
before launching a complete dataset.  Full CUDA inference remains the evidence
gate for any newly reproduced paper value.
