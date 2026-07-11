# InstructPix2Pix SECOND Baseline

This wrapper evaluates the official `timbrooks/instruct-pix2pix` checkpoint on
SECOND using the same shared one-class-per-direction selection JSONL as Vistar
and RCDGen. Model inputs are only the source RGB image and an edit instruction;
the saved `cond_mask*` files are never passed to the model.

```bash
pip install -r requirements-instructpix2pix.txt
bash scripts/bootstrap_instructpix2pix.sh
CUDA_VISIBLE_DEVICES=0 python tools/check_instructpix2pix_deps.py
```

The bootstrap automatically downloads/resumes the Hugging Face snapshot into
`/root/data/weight/instructpix2pix/instruct-pix2pix`.

Two-GPU smoke test and full run:

```bash
SECOND_ROOT=/root/data/second_dataset MAX_SAMPLES=1 bash run_bash/instructpix2pix_second_gen.bash
SECOND_ROOT=/root/data/second_dataset bash run_bash/instructpix2pix_second_gen.bash
```

For one GPU, use `GPU_IDS=0 NPROC_PER_NODE=1`. Both temporal directions are
enabled by default. Outputs follow the Vistar SECOND generation layout:
`source_rgb`, `cond_mask`, `cond_mask_official`, `cond_mask_ids`, `gt_rgb`,
`pred_rgb`, `absdiff`, and `prompts`.
