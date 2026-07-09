# InstanceDiffusion Baseline

This folder contains a Vistar/UniGen wrapper for InstanceDiffusion:

> InstanceDiffusion: Instance-level Control for Image Generation, CVPR 2024.

Official code: https://github.com/frank-xwang/InstanceDiffusion

The wrapper uses the diffusers port described by the official repository:
https://huggingface.co/kyeongry/instancediffusion_sd15

## What This Wrapper Does

The wrapper converts a Vistar-style generation eval directory:

```text
eval_dir/
  cond_mask/*_cond_mask.png
  gt_rgb/*_gt_rgb.png
```

into InstanceDiffusion inputs:

```text
output_dir/instancediffusion_inputs/
  sample.json
```

Each JSON contains a global remote-sensing caption and one open-set phrase per
LoveDA foreground class found in the semantic mask. Because the diffusers port
accepts bounding boxes, not mask RLEs, this wrapper converts each semantic class
region into one normalized `xyxy` box by default. Background is skipped unless
`INCLUDE_BACKGROUND=1` is set.

Inference writes the same output layout as other baselines:

```text
output_dir/
  pred_rgb/
  pred_rgb_native/
  cond_mask/
  gt_rgb/
  instancediffusion_inputs/
  manifest_resolved.jsonl
```

## Required Weights

InstanceDiffusion needs:

- the diffusers fork branch with `StableDiffusionINSTDIFFPipeline`
- the HuggingFace model `kyeongry/instancediffusion_sd15`

The one-command bash clones the diffusers fork under
`third_party/diffusers-instancediffusion` by default. The model is loaded through
HuggingFace/diffusers at inference time.

## Run LoveDA Mask-to-RGB

```bash
cd /root/code/vistarmodels
bash scripts/bootstrap_instancediffusion.sh
pip install -r requirements-instancediffusion.txt
python tools/check_instancediffusion_deps.py
MAX_SAMPLES=5 bash run_bash/instancediffusion_loveda_gen.bash
bash run_bash/instancediffusion_loveda_gen.bash
```

Resume is enabled by default. Existing valid
`pred_rgb/<name>_pred_rgb.png` files are skipped unless `OVERWRITE=1` is set.
If an interrupted run leaves a corrupt or wrong-size prediction, it is
regenerated automatically.

Default inference runs one image at a time (`BATCH_SIZE=1`) at
`RESOLUTION=512` and writes `EVAL_SIZE=512` outputs. If memory is tight, use:

```bash
RESOLUTION=256 EVAL_SIZE=256 bash run_bash/instancediffusion_loveda_gen.bash
```

By default, the script uses physical GPU 1:

```bash
CUDA_VISIBLE_DEVICES=1
```

Override it if needed:

```bash
CUDA_VISIBLE_DEVICES=0 bash run_bash/instancediffusion_loveda_gen.bash
```

For lower peak GPU memory, try:

```bash
CPU_OFFLOAD=1 bash run_bash/instancediffusion_loveda_gen.bash
```

## Notes

This is a box-conditioned InstanceDiffusion baseline. It is open-set because
each LoveDA region is passed as a free-form text phrase, but it is not a
pixel-accurate semantic-mask conditioning model. Report it as
InstanceDiffusion-box when comparing against mask-conditioned baselines.
