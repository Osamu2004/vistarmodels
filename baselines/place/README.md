# PLACE Baseline

This folder contains a Vistar/UniGen wrapper for PLACE:

> PLACE: Adaptive Layout-Semantic Fusion for Semantic Image Synthesis, CVPR
> 2024.

Official code: https://github.com/cszy98/PLACE

## What This Wrapper Does

The wrapper converts a Vistar-style generation eval directory:

```text
eval_dir/
  cond_mask/*_cond_mask.png
  gt_rgb/*_gt_rgb.png
```

into PLACE inputs:

```text
output_dir/place_inputs/
  sample_label.png
  sample.json
```

Each LoveDA mask color is converted to a 1-based semantic label id. Label `0`
is ignored as background, matching PLACE's ADE20K preprocessing. The wrapper
then tokenizes the LoveDA class descriptions and passes both the semantic label
map and token-to-class ids into the official PLACE LDM model.

Inference writes the same output layout as other baselines:

```text
output_dir/
  pred_rgb/
  pred_rgb_native/
  cond_mask/
  gt_rgb/
  place_inputs/
  manifest_resolved.jsonl
```

## Required Weights

PLACE needs:

- official PLACE source code under `third_party/PLACE`
- a PLACE checkpoint, default path:
  `/root/data/weight/place/coco_best.ckpt`

The official pretrained models are linked from the PLACE README:
https://drive.google.com/drive/folders/1b5pC52hasLwm1gOkc9LmdIyxZjrdlNWC

Download either `coco_best.ckpt` or `ade20k_best.ckpt` and set `PLACE_CKPT` if
you store it somewhere else.

## Run LoveDA Mask-to-RGB

```bash
cd /root/code/vistarmodels
bash scripts/bootstrap_place.sh
pip install -r requirements-place.txt
python tools/check_place_deps.py
MAX_SAMPLES=5 bash run_bash/place_loveda_gen.bash
bash run_bash/place_loveda_gen.bash
```

Resume is enabled by default. Existing valid
`pred_rgb/<name>_pred_rgb.png` files are skipped unless `OVERWRITE=1` is set.
If an interrupted run leaves a corrupt or wrong-size prediction, it is
regenerated automatically.

PLACE has hard-coded 512-pixel semantic attention in the official code, so
`RESOLUTION` must stay `512`. You can still resize metric outputs with
`EVAL_SIZE`, for example:

```bash
EVAL_SIZE=256 bash run_bash/place_loveda_gen.bash
```

By default, the script uses physical GPU 1:

```bash
CUDA_VISIBLE_DEVICES=1
```

Override it if needed:

```bash
CUDA_VISIBLE_DEVICES=0 bash run_bash/place_loveda_gen.bash
```

## Notes

PLACE is a semantic image synthesis baseline, not a free-form global prompt
model. The text condition is the set of class-name tokens associated with the
semantic mask regions. For LoveDA, the wrapper uses descriptions for buildings,
roads, water, barren land, forest, and agricultural fields.

The wrapper defaults to ignoring the black background class. Set
`INCLUDE_BACKGROUND=1` to condition on it as an explicit region.
