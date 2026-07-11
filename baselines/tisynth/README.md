# TISynth LoveDA Baseline

This directory adapts the vendored official
[dongrunmin/TISynth](https://github.com/dongrunmin/TISynth) implementation under
`third_party/TISynth` to
Vistar's ordinary LoveDA semantic-mask-to-image comparison at exactly
512x512.

The official source is committed directly into this repository at upstream
revision `688cda0597b1550cb32bbae0469b8a0a900501c0`. The bootstrap script only
checks the local copy; it performs no network clone, fetch, or pull.

TISynth is reference-conditioned. Its inference inputs are:

1. a colored LoveDA semantic mask;
2. a per-sample class prompt derived from that mask;
3. an RGB reference image selected deterministically from an explicitly
   supplied reference pool.

The paired GT is saved only for evaluation and is never passed to the model.
The manifest records the exact reference image and seed for every sample.

## Checkpoint protocol

The authors' public Google Drive archive contains `controlnet1.5.ckpt` and a
`GID_model.ckpt`. Vistar's comparison protocol evaluates externally trained
models directly on LoveDA, so the default inference checkpoint is the official
GID-trained `GID_model.ckpt`; no LoveDA fine-tuning is performed. The generated
run directory explicitly contains `gid_zeroshot` to preserve this provenance.
`controlnet1.5.ckpt` is only the initialization weight used when training a new
TISynth model and is not the inference checkpoint.

Official weights archive:
https://drive.google.com/file/d/15i-beG-7b5lLL_pJXSI7mVT4zokNBJib/view

## Optional LoveDA training (not used by the zero-shot table protocol)

Prepare disjoint Vistar-style training and validation directories, each with
`cond_mask` and `gt_rgb`, then initialize from the official
`controlnet1.5.ckpt`:

```bash
TRAIN_SOURCE_DIR=/root/data/experiment/loveda_train_512 \
VAL_SOURCE_DIR=/root/data/experiment/loveda_val_512 \
TISYNTH_PRETRAIN=/root/data/weight/TISynth/controlnet1.5.ckpt \
GPU_IDS=0, \
MAX_STEPS=100000 \
bash run_bash/tisynth_loveda_train.bash
```

The default saves every 10,000 steps. Training manifests use absolute paths,
derive class prompts from the official LoveDA palette, and reject an identical
train/validation source directory. Resume by setting `RESUME` to the run folder
or a checkpoint accepted by the official `main.py`.

## Environment

Exact reproduction should use the official `tisynth.yml` (Python 3.8,
PyTorch 2.0.1, CUDA 11.8). The smaller requirements file only lists the direct
inference dependencies and deliberately leaves CUDA-matched PyTorch wheels to
the user.

```bash
cd /root/code/vistarmodels
bash scripts/bootstrap_tisynth.sh  # verifies the vendored local source only
conda env create -f third_party/TISynth/tisynth.yml
conda activate tisynth
python tools/check_tisynth_deps.py
```

## 512x512 LoveDA inference

The wrapper reads the official dataset tree directly using the same defaults as
Vistar: `LOVEDA_ROOT=/root/data/LoveDA`, `SPLITS=train,val`, and
`DOMAINS=both`. It supports
`LoveDA/{Train,Val}/{Urban,Rural}/{images_png,masks_png}`, converts the official
raw IDs `1..7` to class IDs `0..6`, and constructs the official colored mask at
512x512. No pre-generated Vistar evaluation directory is required.

TISynth also requires a reference RGB image. `REFERENCE_DIR` defaults to
`$LOVEDA_ROOT/Train`; a reference is selected by a stable SHA-256 mapping and
the exact paired target is always excluded. Every choice is recorded in the
manifest.

```bash
cd /root/code/vistarmodels

# Five-sample smoke test with the official GID-trained checkpoint.
LOVEDA_ROOT=/root/data/LoveDA \
CUDA_VISIBLE_DEVICES=0 \
MAX_SAMPLES=5 \
bash run_bash/tisynth_loveda_gen.bash

# Full run: resume is automatic.
LOVEDA_ROOT=/root/data/LoveDA \
CUDA_VISIBLE_DEVICES=0 \
bash run_bash/tisynth_loveda_gen.bash
```

Under this protocol, omit `TISYNTH_CKPT`; it defaults to
`/root/data/weight/TISynth/GID_model.ckpt`. Set it explicitly only when running
a separately named fine-tuned experiment.

Defaults match the official inference implementation's effective settings:
512x512, DDIM 50 steps, CFG 9, strength 1, eta 0. The wrapper fixes three
official-script issues relevant to fair evaluation: the parsed CFG value was
ignored in favor of a hard-coded 9; batch resume checked only the last sample;
and output was JPEG. This adapter uses the requested CFG, checks every sample,
and saves lossless PNGs.

Output contract:

```text
output_dir/
  cond_mask/*_cond_mask.png
  gt_rgb/*_gt_rgb.png
  reference_rgb/*_reference_rgb.png
  pred_rgb/*_pred_rgb.png
  pred_rgb_native/*_pred_rgb_512.png
  manifest_loveda_tisynth.jsonl
  manifest_loveda_tisynth.protocol.json
  manifest_resolved.jsonl
  run_config.json
```

## Unified metrics

Run metrics in the main Vistar environment after generation:

```bash
conda activate vistar_flux
cd /root/code/vistar
GPU_ID=0 INPUT_SIZE=512 \
bash run_bash/seg/compute_saved_loveda_gen_metrics.bash \
  /root/data/experiment/tisynth_gid_zeroshot_loveda_mask_to_rgb_gen_resize512_steps50_cfg9p0_seed0_refseed0
```

The resulting `metrics.json` includes paired PSNR/SSIM/LPIPS and the same
distribution metrics used by Vistar (`loveda_fd6` and `dinov3sat_fid`, plus
CMMD when enabled).
