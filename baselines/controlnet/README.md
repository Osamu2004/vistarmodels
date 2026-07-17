# SD 1.5 ControlNet for SECOND

This baseline trains a ControlNet from the local Stable Diffusion 1.5 snapshot
on both directions of SECOND.  Training uses the target-side directional
semantic change mask and a class-aware English prompt.  The default inference
protocol then uses the source-time image as the img2img initialization, the
target-side directional semantic change mask as the ControlNet input, and the
same class-aware text.  Its condition is therefore **source image + change mask
+ text**.

For example, a T1-to-T2 row containing building and playground changes uses:

```text
Generate a realistic 256 by 256 post-change remote sensing image. The
target-side semantic change mask contains changed building and changed
playground; unchanged pixels are labeled unchanged.
```

The reverse row uses `pre-change` and derives its changed-class list from the
reverse target mask.  The class vocabulary is `inland water`, `bare land`,
`grass`, `forest`, `building`, and `playground`.

For T1-to-T2 generation, T1 is the source image and the post-side directional
mask is the control input.  For T2-to-T1 generation, T2 is the source image and
the pre-side directional mask is the control input.  The existing trained
ControlNet can be reused; source conditioning is introduced by the SD 1.5
ControlNet img2img inference pipeline rather than by changing its trainable
ControlNet weights.

## One-command training

The default run is intentionally one GPU with batch size 2, no gradient
accumulation, two DataLoader workers, bf16, 256x256 inputs, and 100K optimizer
steps.  It first builds path-only train/test manifests, verifies the local
weights and data, runs a two-step plus resume smoke test, and then starts or
resumes the full run.

```bash
cd /root/code/vistarmodels

SECOND_ROOT=/root/data/second_dataset \
BASE_MODEL=/root/data/weight/stable-diffusion-v1-5 \
GPU_IDS=0 \
NPROC_PER_NODE=1 \
PER_GPU_BATCH=2 \
bash run_bash/controlnet_second_oneclick.bash
```

If the active environment lacks the pinned packages, add `INSTALL_DEPS=1` to
that command or install them once with:

```bash
python -m pip install -r requirements-controlnet.txt
```

The default full output is:

```text
/root/data/experiment/controlnet_sd15_second_mask_text_256_1gpu_bs2_seed42
```

`RESUME=auto` is the default.  To continue the full run without repeating the
smoke test:

```bash
cd /root/code/vistarmodels

GPU_IDS=0 NPROC_PER_NODE=1 PER_GPU_BATCH=2 \
RUN_SMOKE=0 RUN_FULL=1 RESUME=auto \
bash run_bash/controlnet_second_oneclick.bash
```

Checkpoints are full Accelerate states under `checkpoint-XXXXXXX`; the final
directory also contains a Diffusers `save_pretrained` ControlNet.

## Generation and common SECOND metrics

The inference exporter writes `source_rgb`, `cond_mask`,
`cond_mask_official`, `cond_mask_ids`, `gt_rgb`, `pred_rgb`, `absdiff`, and
`prompts`, plus JSONL/config metadata.  This is the same saved-output contract
used by the other SECOND generation baselines.

```bash
cd /root/code/vistarmodels

CONTROLNET=/root/data/experiment/controlnet_sd15_second_mask_text_256_1gpu_bs2_seed42 \
BASE_MODEL=/root/data/weight/stable-diffusion-v1-5 \
GPU_IDS=0,1 NPROC_PER_NODE=2 \
PIPELINE_MODE=source_img2img STRENGTH=0.8 \
COMPUTE_METRICS=1 \
bash run_bash/controlnet_second_gen.bash
```

For single-GPU inference, set `GPU_IDS=0 NPROC_PER_NODE=1`.  The common metric
stage reports mFD6, FID-Sat, PSNR, SSIM, LPIPS, CMMD, and frozen directional
change-consistency metrics when the VISTAR evaluator and detector checkpoint
are available.  The default output is isolated at:

```text
/root/data/experiment/controlnet_sd15_second_source_mask_text_test_256_steps50_cfg7p5_strength0p8_seed42
```

`STRENGTH=0.8` retains source-scene information while allowing the requested
change.  A value near 1.0 injects almost pure noise and consequently weakens
source-image preservation.

The previous mask-and-text-only result remains reproducible with an explicit
legacy mode and a separate output directory:

```bash
PIPELINE_MODE=mask_text2img \
OUTPUT_DIR=/root/data/experiment/controlnet_sd15_second_mask_text_test_256_steps50_cfg7p5_seed42 \
bash run_bash/controlnet_second_gen.bash
```

Do not mix metrics from the two protocols.  For new paper comparisons, label
the condition as **source image + change mask + text**.
