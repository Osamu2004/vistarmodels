# RCDGen Baseline

This wrapper runs the official WACV 2026 RCDGen model on SECOND in the Vistar
result layout. RCDGen conditions on a source image and one category prompt; it
does not receive the ground-truth spatial change mask. It jointly generates a
post-change RGB image and a binary change mask.

Install the CUDA-matched PyTorch build first, then:

```bash
pip install -r requirements-rcdgen.txt
bash scripts/bootstrap_rcdgen.sh
python tools/check_rcdgen_deps.py
```

The official custom pipeline is vendored under
`third_party/referring_change_detection/RCDGen`. The bootstrap installs it into
Diffusers 0.31.0 and downloads/resumes the complete Hugging Face snapshot from
`yilmazkorkmaz/RCDGen` into `/root/data/weight/rcdgen/RCDGen`. Override
`RCDGEN_MODEL_DIR` to use another shared weight directory. Set
`RCDGEN_DOWNLOAD_WEIGHTS=0` only when the snapshot already exists.

Smoke test and full run:

```bash
SECOND_ROOT=/root/data/SECOND MAX_SAMPLES=1 bash run_bash/rcdgen_second_gen.bash
SECOND_ROOT=/root/data/SECOND bash run_bash/rcdgen_second_gen.bash
```

The RCDGen builder is independent of DreamCD and accepts the official
`test/A`, `test/B`, and `test/gt` layout. It does not require DreamCD dense
`mask_A`/`mask_B` pseudo masks. The default direction is `t1_to_t2`, matching
the official RCDGen pre-change-to-post-change protocol. Reverse inference is
available only when a directional `label1` folder exists.

By default, each SECOND record selects one changed target category using a
record-local seeded RNG. It therefore produces exactly one prediction per
sample and direction, keeps every GT image once, and is directly usable for
FID and paired metrics. Set `CATEGORY_POLICY=all` only for the expanded
per-category qualitative protocol. Final outputs follow the Vistar SECOND
layout and additionally contain `pred_change_mask/`.
The `cond_mask*` files are retained only for evaluation/alignment and are never
passed into RCDGen. This protocol should not be described as spatially
mask-conditioned generation.
