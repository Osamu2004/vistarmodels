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
SECOND_ROOT=/root/data/second_dataset \
CLASS_SELECTION_FILE=/root/data/experiment/protocols/second_test_oneclass_targetmask_both_resize256_seed42_labelpairauto.jsonl \
MAX_SAMPLES=1 bash run_bash/rcdgen_second_gen.bash
SECOND_ROOT=/root/data/second_dataset \
CLASS_SELECTION_FILE=/root/data/experiment/protocols/second_test_oneclass_targetmask_both_resize256_seed42_labelpairauto.jsonl \
bash run_bash/rcdgen_second_gen.bash
```

RCDGen uses the exact `CLASS_SELECTION_FILE` created by Vistar's
`eval_flux2_second_oneclass_binarymask_gen.bash`; it never makes an independent
random class choice. The builder requires official directional `label1` and
`label2` folders, revalidates each selected class after the record's original
label preprocessing and resize, and materializes the matching selected-class
masks next to `manifest.jsonl`. Thus `name + direction` has the same target
class for Flux2, RCDGen, and future Vistar-model baselines. A reduced `A/B/gt`
copy cannot be used for this shared directional semantic protocol.

The shared protocol produces exactly one prediction per sample and direction,
keeps every GT image once, and is directly usable for FID and paired metrics.
Final outputs follow the Vistar SECOND layout and additionally contain
`pred_change_mask/`.
The default launcher uses two data-parallel GPUs (`GPU_IDS=0,1`,
`NPROC_PER_NODE=2`). Each GPU owns disjoint manifest rows; it does not split a
single diffusion model across cards. Override both variables together for a
different GPU count.
The `cond_mask*` files are retained only for evaluation/alignment and are never
passed into RCDGen. This protocol should not be described as spatially
mask-conditioned generation.
