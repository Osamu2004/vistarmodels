# CRS-Diff Baseline

This wrapper runs the official CRS-Diff inference code on Vistar-style
mask-to-RGB generation samples.

Official CRS-Diff repository:

- <https://github.com/Sonettoo/CRS-Diff>
- Paper: *CRS-Diff: Controllable Remote Sensing Image Generation with Diffusion
  Model*, IEEE TGRS 2024.

## What CRS-Diff Can Compare Fairly

CRS-Diff is a remote-sensing controllable image generation model. Its released
code supports inference, not training. The model accepts:

- text prompt,
- local controls: MLSD, HED, sketch, road, depth, segmentation,
- global controls: content and metadata embeddings.

For Vistar, the fairest direct comparison is:

```text
semantic/change mask RGB -> remote-sensing RGB image
```

using the CRS-Diff `seg` local-control slot.

CRS-Diff is not a native bidirectional change-generation model. For SECOND
change generation, it can be used as an external mask-conditioned generation
baseline by feeding the class-change mask to the `seg` slot. It should not be
claimed as an equally conditioned baseline unless source-image/content
conditioning is also made comparable.

## Bootstrap

```bash
cd /root/code/vistarmodels
pip install -r requirements.txt
python tools/check_crsdiff_deps.py
bash scripts/bootstrap_crsdiff.sh
```

Put the official CRS-Diff checkpoint at:

```text
/root/data/weight/crsdiff/last.ckpt
```

or pass `--ckpt` manually.

The official CRS-Diff config hard-codes the authors' local CLIP path. This
wrapper overrides it with:

```text
openai/clip-vit-large-patch14
```

If the server cannot download from HuggingFace, download it manually and run:

```bash
CRSDIFF_CLIP_VERSION=/root/data/weight/clip-vit-large-patch14 \
bash run_bash/crsdiff_loveda_gen.bash
```

## Build A Manifest From An Existing Vistar Eval Directory

The easiest fair setup is to reuse the exact `cond_mask` and `gt_rgb` images
saved by a Vistar eval run. This guarantees the same split, sample count,
resize, palette, and direction names.

```bash
python tools/build_manifest_from_vistar_eval.py \
  --eval_dir /root/data/experiment/eval_flux2_second_test_change_mask_to_rgb_gen_resize256_gen50_cfg2p0_ema_full_step150000_ema_both_official_palette_2gpu \
  --output /root/data/experiment/crsdiff_second_manifest.jsonl \
  --prompt "A high-resolution remote-sensing image matching the given semantic change mask."
```

## Run CRS-Diff

```bash
python baselines/crsdiff/run_crsdiff_manifest.py \
  --crsdiff_root third_party/CRS-Diff \
  --ckpt /root/data/weight/crsdiff/last.ckpt \
  --manifest /root/data/experiment/crsdiff_second_manifest.jsonl \
  --output_dir /root/data/experiment/crsdiff_second_gen \
  --condition_slot seg \
  --resolution 512 \
  --eval_size 512 \
  --batch_size 2 \
  --ddim_steps 50 \
  --scale 7.5 \
  --seed 0
```

Outputs are saved in a Vistar-like layout:

```text
pred_rgb/*_pred_rgb.png
pred_rgb_native/*_pred_rgb_512.png
gt_rgb/*_gt_rgb.png
cond_mask/*_cond_mask.png
manifest_resolved.jsonl
```

The `pred_rgb` images are resized to `eval_size`, so they can be evaluated with
the same image-distribution metrics as Vistar outputs.

Existing `pred_rgb/<name>_pred_rgb.png` files are skipped unless `--overwrite`
is passed. The one-command bash defaults to `BATCH_SIZE=2` and `OVERWRITE=0`,
so interrupted runs can be resumed by rerunning the same command.

## One-Command LoveDA Train+Val Generation

If you already have the Vistar LoveDA generation eval output, run:

```bash
cd /root/code/vistarmodels
bash run_bash/crsdiff_loveda_gen.bash
```

Default input is the merged Vistar LoveDA train+val generation-eval directory:

```text
/root/data/experiment/eval_loveda_gen_gen_only_step300000
```

Default output:

```text
/root/data/experiment/crsdiff_loveda_val_mask_to_rgb_gen_resize512_steps50_scale7p5_seed0
```

The output path intentionally keeps its legacy `val` name so an existing
val-only CRS-Diff run can be continued without copying or regenerating its
predictions. LoveDA val samples keep their old names; train samples are prefixed
with `train_`. With `OVERWRITE=0`, valid val predictions are skipped and only
missing train predictions are generated.

For the existing CRS-Diff result, run the dedicated continuation script:

```bash
bash run_bash/crsdiff_loveda_train_val_continue.bash
```

By default the script verifies that the source contains 2,522 train and 1,669
val samples. Set `VERIFY_SAMPLE_COUNTS=0` only for a partial smoke test.

Small smoke test:

```bash
MAX_SAMPLES=5 bash run_bash/crsdiff_loveda_gen.bash
```

Use a larger or smaller generation batch:

```bash
BATCH_SIZE=4 bash run_bash/crsdiff_loveda_gen.bash
```

Override the Vistar eval source:

```bash
VISTAR_EVAL_DIR=/root/data/experiment/your_loveda_eval_dir \
OUTPUT_DIR=/root/data/experiment/crsdiff_loveda_your_run \
bash run_bash/crsdiff_loveda_gen.bash
```
