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

## Boundary WFm from saved predictions

The CPU-only scorer `tools/score_rcdnet_boundary.py` reads this adapter's
`pred_mask/<stem>_pred_mask.png`, `gt_mask/<stem>.png`, `metrics.json`, and
`per_image_metrics.jsonl`. It applies the shared IDGBR-style 3px boundary WFm
to each stitched, native-resolution mask. No model, Torch, or CUDA is used.
Dependencies are NumPy, SciPy, and Pillow. The shared metric uses OpenCV when
available and its existing SciPy fallback otherwise.

```bash
python tools/score_rcdnet_boundary.py --eval_dir /path/to/your/rcdnet_output
```

The default requires all 128 LEVIR-CD test masks at 1024x1024, matching sample
names, original per-image/global counts, and the saved IoU/F1. Source SHA256
values are checked before and after scoring. Outputs are separate files in
`<eval_dir>/boundary_wfm_3px/{paper_summary.json,per_image_boundary.json}`;
original predictions and metrics are retained. The paper value is
`scores.wfm_3px_percent`. Dataset aggregation averages edge and non-edge
scores separately over GT-present images before taking the smaller positive
class mean; it is not an average of per-image minima or tile scores.

## Recorded saved-mask result (2026-11-09)

The user completed all 128 native LEVIR-CD test masks in the `rcdnet` environment, from `/root/data/experiment/rcdnet_levircd_test_native1024_tile512`. The saved-mask scorer reported changed-class IoU **46.03432144395981%** and 3px WFm **28.45222020216903%**, with all source-count, IoU/F1, and SHA gates passing. The terminal receipt is [archived here](receipts/wsl_levircd_wfm_20261109.json); the underlying WSL JSON and predictions were not independently read or rescored on this Mac.

These are adapter results. Original-paper Table 1 reports LEVIR-CD IoU 53.98 for real-data training and 60.21 for synthetic pretraining; it does not report this WFm. The manuscript uses the matched measured pair 46.03/28.45 rather than combining published IoU with this saved-mask WFm.

## Campus rerun

The separately authorized November9 campus rerun is tracked in [the experiment node](../../experiments/stage1_rcdnet_campus_20261109/README.md). Its preparation state and later measured results remain separate from the saved WSL receipt above.

校园本次结果：整图缩放512 IoU40.117765%/WFm16.586877%；原图512分块 IoU46.035337%/WFm28.453593%，各128/128、0失败、退出0。分块与旧WSL回执几乎一致，完整证据见[校园总结](../../experiments/stage1_rcdnet_campus_20261109/SUMMARY.md)。
