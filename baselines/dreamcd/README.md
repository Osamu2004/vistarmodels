# DreamCD Baseline

This wrapper runs the official DreamCD SECOND synthesis code in the same
manifest-to-output style used by the other `vistarmodels` baselines.

Official DreamCD repository:

- <https://github.com/tangkai-RS/DreamCD>
- Weights: <https://huggingface.co/tangkaii/DreamCD>
- Paper: *DreamCD: A Change-Label-Free Framework for Change Detection via a
  Weakly Conditional Semantic Diffusion Model in Optical VHR Imagery*, JAG 2026.

## What This Baseline Is

DreamCD is a closed-set, SECOND/LsSCD-style change image synthesis model. It is
not open-set and it is not a text-driven open-vocabulary model. Its inference
condition is:

```text
source image A + source semantic mask A + target semantic mask B + binary change mask -> synthetic image B
```

The official code can also use the real target image B for AdaIN style transfer.
For fair comparison, this wrapper disables that path by default: target B is not
used as an inference condition or style reference. If target B is present in the
manifest, it is retained only as `gt_rgb` for metric computation; the runtime
DreamCD input receives source A as the inactive image-B placeholder.

Set `WITH_ADAIN=1` in the one-command runner, or pass `--with_adain` directly,
to explicitly restore the official-demo behavior that uses real target B style
information. AdaIN and non-AdaIN runs use separate default output directories.

For a fair report, describe DreamCD as a SECOND-trained closed-set baseline with
semantic-mask and binary-change-mask conditioning.

## Bootstrap

```bash
cd /root/code/vistarmodels
pip install -r requirements-dreamcd.txt
bash scripts/bootstrap_dreamcd.sh
python tools/check_dreamcd_deps.py
```

`bootstrap_dreamcd.sh` clones the official DreamCD repo and downloads the
official SECOND weights from HuggingFace by default:

```text
third_party/DreamCD/checkpoints/second/vqvae.ckpt
third_party/DreamCD/checkpoints/second/ldm.ckpt
```

Use `DREAMCD_DOWNLOAD_WEIGHTS=0` to skip downloading, or pass
`DREAMCD_VQVAE_CKPT` and `DREAMCD_CKPT` to use custom paths.

DreamCD's official environment pins an old CUDA 11.1 PyTorch stack. In practice,
use a separate conda environment and let `tools/check_dreamcd_deps.py` tell you
what is missing.

## Build A SECOND Manifest

The manifest builder recognizes the official DreamCD example layout:

```text
img_A/img_B/mask_A/mask_B/bcd_mask
```

and common SECOND-style names:

```text
im1/im2/label1/label2/change
```

```bash
python tools/build_dreamcd_second_manifest.py \
  --second_root /root/data/SECOND \
  --split test \
  --direction t1_to_t2 \
  --allow_missing_change_mask \
  --output /root/data/experiment/dreamcd_second_test_manifest.jsonl
```

If no binary change mask exists, `--allow_missing_change_mask` lets the runner
derive it from `source_mask != target_mask`.

## Run DreamCD

```bash
python baselines/dreamcd/run_dreamcd_manifest.py \
  --dreamcd_root third_party/DreamCD \
  --manifest /root/data/experiment/dreamcd_second_test_manifest.jsonl \
  --output_dir /root/data/experiment/dreamcd_second_test_gen \
  --ckpt third_party/DreamCD/checkpoints/second/ldm.ckpt \
  --vqvae_ckpt third_party/DreamCD/checkpoints/second/vqvae.ckpt \
  --resolution 256 \
  --eval_size 256 \
  --batch_size 16 \
  --ddim_steps 200 \
  --seed 2025
```

One-command SECOND run:

```bash
SECOND_ROOT=/root/data/SECOND \
SPLIT=test \
MAX_SAMPLES=5 \
bash run_bash/dreamcd_second_gen.bash

SECOND_ROOT=/root/data/SECOND \
SPLIT=test \
bash run_bash/dreamcd_second_gen.bash
```

Existing `pred_rgb/<name>_pred_rgb.png` files are skipped unless `OVERWRITE=1`
is set, so interrupted runs can be resumed by rerunning the same command.
The default output directory includes `noadain`; an explicit `WITH_ADAIN=1` run
uses an `adain` directory instead.

Outputs are saved in a Vistar-like layout:

```text
pred_rgb/*_pred_rgb.png
pred_rgb_native/*_pred_rgb_256.png
source_rgb/*_source_rgb.png
gt_rgb/*_gt_rgb.png
cond_mask/*_cond_mask.png
source_mask/*_source_mask.png
target_mask/*_target_mask.png
change_mask/*_change_mask.png
runtime/dreamcd_sample_list.txt
manifest_resolved.jsonl
```
