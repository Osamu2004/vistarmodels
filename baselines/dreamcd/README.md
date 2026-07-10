# DreamCD Baseline

This wrapper runs the official DreamCD SECOND synthesis code in the same
manifest-to-output style used by the other `vistarmodels` baselines.

Official DreamCD repository:

- <https://github.com/tangkai-RS/DreamCD>
- Weights: <https://huggingface.co/tangkaii/DreamCD>
- Paper: *DreamCD: A Change-Label-Free Framework for Change Detection via a
  Weakly Conditional Semantic Diffusion Model in Optical VHR Imagery*, JAG 2026.

The required official inference source is vendored under
`third_party/DreamCD` at upstream commit
`d4750ff6f7d35fe9640059d7b9cdfe6902fcf9c5`. A fresh clone of `vistarmodels`
therefore does not need a separate DreamCD source clone.

## What This Baseline Is

DreamCD is a closed-set, SECOND/LsSCD-style change image synthesis model. It is
not open-set and it is not a text-driven open-vocabulary model. Its inference
condition is:

```text
source image A + source semantic mask A + target semantic mask B + binary change mask -> synthetic image B
```

The official code can also use the real target image B for AdaIN style transfer.
For fair comparison, this wrapper disables that path by default: target B is not
used as an inference condition or style reference. The target B path remains
required only to populate Vistar's `gt_rgb` directory for metric computation;
the runtime DreamCD input receives source A as the inactive image-B placeholder,
so real target B is isolated from the model input by default.

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

`bootstrap_dreamcd.sh` verifies the vendored official source and downloads the
official SECOND weights from HuggingFace by default. It retains a clone fallback
only for older checkouts where `third_party/DreamCD` is absent:

```text
/root/data/weight/dreamcd/second/vqvae.ckpt
/root/data/weight/dreamcd/second/ldm.ckpt
```

Use `DREAMCD_DOWNLOAD_WEIGHTS=0` to skip downloading, or pass
`DREAMCD_VQVAE_CKPT` and `DREAMCD_CKPT` to use custom paths. On Windows, the
default folder is visible at:

```text
\\wsl.localhost\Ubuntu-22.04\root\data\weight\dreamcd\second\
```

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
  --ckpt /root/data/weight/dreamcd/second/ldm.ckpt \
  --vqvae_ckpt /root/data/weight/dreamcd/second/vqvae.ckpt \
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

The one-command runner defaults to the Vistar evaluation protocol:
`SPLIT=test` and `DIRECTION=both`. Override either variable only for a targeted
single-split or single-direction run.

Existing `pred_rgb/<name>_pred_rgb.png` files are skipped unless `OVERWRITE=1`
is set, so interrupted runs can be resumed by rerunning the same command.
The default output directory includes `noadain_vistar_layout`; an explicit
`WITH_ADAIN=1` run uses `adain_vistar_layout` instead.

Final outputs use the same SECOND generation directory contract as
`vistar/eval_flux2_second_gen.py`:

```text
output_dir/
  source_rgb/*_source_rgb.png
  cond_mask/*_cond_mask.png
  cond_mask_official/*_cond_mask_official.png
  cond_mask_ids/*_cond_mask_ids.png
  gt_rgb/*_gt_rgb.png
  pred_rgb/*_pred_rgb.png
  absdiff/*_absdiff.png
  prompts/<name>.txt
  class_map.json
  prompt_<direction>_raw.txt
  prompt_<direction>_effective.txt
  prompts.jsonl
```

`cond_mask*` stores the direction-specific target-class change condition, with
ID 0 reserved for unchanged pixels, matching Vistar's SECOND generation output
semantics. DreamCD-only native predictions, masks, patched config, CSV, and
preview files are kept in a temporary working directory outside `output_dir`.
Set `RUNTIME_DIR=/path/to/workdir` only when those internal artifacts need to be
retained for debugging. The generated input manifest is also stored adjacent to
the result directory rather than inside it.
