# RCDNet on LEVIR-CD

This adapter runs the authors' official RCDNet architecture on the official
LEVIR-CD layout:

```text
/root/data/LEVIR-CD/test/
├── A/
├── B/
└── label/
```

RCDNet receives each registered A/B pair and the text query `building`. The
released SECOND checkpoint is evaluated cross-domain. Native 1024x1024 images
are processed as non-overlapping 512x512 tiles, matching the checkpoint's
training input size, and stitched before metrics or images are written.

## Important checkpoint distinction

The authors currently publish `SECOND-model.safetensors`, trained only on real
SECOND data. This is the default used here and corresponds to the paper's
non-synthetic cross-domain setting (paper value: 53.98 changed-class IoU on
LEVIR-CD). The stronger synthetic-pretrained checkpoint used for the paper's
60.21 result is not publicly released as of 2026-07-21. If it becomes
available, set `RCDNET_CHECKPOINT=/path/to/checkpoint.safetensors`; no code
change is required.

## Setup and inference

```bash
python -m pip install -r requirements-rcdnet.txt
bash scripts/bootstrap_rcdnet.sh
python tools/check_rcdnet_deps.py

MAX_SAMPLES=2 bash run_bash/rcdnet_levircd.bash
bash run_bash/rcdnet_levircd.bash
```

The bootstrap pins official source revision
`0966e96ff7075476d77442bbf6623ed5086d52da`, downloads the public SECOND
checkpoint, and builds the official selective-scan CUDA extension.

The public checkpoint also contains 88 legacy tensors: 80 from four unused
`cross_mamba` blocks and eight `norm2` tensors from decoder blocks whose current
forward path uses only `norm1` and `norm3`. These tensors are neither registered
nor executed by the pinned official source. The adapter filters only that exact
release schema and records the removed names under
`checkpoint_load.ignored_legacy_keys` in `metrics.json`; missing active weights
or any other unexpected checkpoint keys still fail by default.

## Outputs

The output is directly usable for qualitative Figure 5 selection:

```text
output_dir/
├── input_A/
├── input_B/
├── gt_mask/
├── pred_mask/       # <stem>_pred_mask.png, binary 0/255 masks
├── pred_rgb/
├── error_map/       # TP white, TN black, FP green, FN red
├── overlay_A/
├── overlay_B/
├── metrics.json
└── per_image_metrics.jsonl
```

`metrics.json` uses the same global-pixel OpenDPR/DynamicEarth equations as
VISTAR's LEVIR-CD evaluator. The prediction filenames are directly accepted by
`vistar/tools/compute_saved_levircd_opendpr_metrics.py`. The output protocol is intentionally explicit:
the model-internal tiling is not claimed to reproduce the paper's unpublished
exact cross-domain preprocessing unless the authors release that detail.
