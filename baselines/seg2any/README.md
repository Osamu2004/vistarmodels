# Seg2Any Baseline

This folder contains a Vistar/UniGen wrapper for Seg2Any:

> Seg2Any: Open-set Segmentation-Mask-to-Image Generation with Precise Shape and
> Semantic Control, NeurIPS 2025.

Official code: https://github.com/0xLDF/Seg2Any

## What This Wrapper Does

The wrapper converts a Vistar-style generation eval directory:

```text
eval_dir/
  cond_mask/*_cond_mask.png
  gt_rgb/*_gt_rgb.png
```

into Seg2Any inputs:

```text
output_dir/seg2any_inputs/
  sample.png
  sample.json
```

Each JSON contains a global remote-sensing caption and one regional caption per
LoveDA mask color. Inference writes the same output layout as other baselines:

```text
output_dir/
  pred_rgb/
  pred_rgb_native/
  cond_mask/
  gt_rgb/
  manifest_resolved.jsonl
```

## Required Weights

Seg2Any needs:

- official Seg2Any source code under `third_party/Seg2Any`
- `black-forest-labs/FLUX.1-dev` or a local FLUX.1-dev folder
- Seg2Any LoRA checkpoint, default path:
  `/root/data/weight/seg2any/sacap_1m/seg2any/checkpoint-20000`

The one-command bash auto-downloads missing weights by default
(`AUTO_DOWNLOAD_WEIGHTS=1`). To download manually:

```bash
huggingface-cli download 0xLDF/Seg2Any \
  --local-dir /root/data/weight/seg2any
hf download black-forest-labs/FLUX.1-dev \
  --local-dir /root/data/weight/flux1/FLUX.1-dev
```

FLUX.1-dev is gated on HuggingFace. Log in or point
`SEG2ANY_FLUX1_MODEL` to a local downloaded folder.

## Run LoveDA Mask-to-RGB

```bash
cd /root/code/vistarmodels
bash scripts/bootstrap_seg2any.sh
pip install -r requirements-seg2any.txt
python tools/check_seg2any_deps.py
MAX_SAMPLES=5 bash run_bash/seg2any_loveda_gen.bash
bash run_bash/seg2any_loveda_gen.bash
```

Resume is enabled by default. Existing valid
`pred_rgb/<name>_pred_rgb.png` files are skipped unless `OVERWRITE=1` is set.
If an interrupted run leaves a corrupt or wrong-size prediction, it is
regenerated automatically.

Default Seg2Any inference runs one image at a time (`BATCH_SIZE=1`) at
`RESOLUTION=256` and writes `EVAL_SIZE=256` outputs. For the older 512-pixel
setting, override with `RESOLUTION=512 EVAL_SIZE=512`.

By default, the script uses physical GPU 1:

```bash
CUDA_VISIBLE_DEVICES=1
```

Override it if needed:

```bash
CUDA_VISIBLE_DEVICES=0 bash run_bash/seg2any_loveda_gen.bash
```

## Fairness Notes

Seg2Any is not a remote-sensing model. It is an open-set mask-to-image model
built on FLUX.1-dev and is useful as a strong general-domain open-set S2I
baseline. It should be reported separately from remote-sensing-specific
baselines such as CRS-Diff and EarthSynth.
