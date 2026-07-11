# FLUX.1 Fill-dev on SECOND

This is a zero-shot FLUX.1 Fill-dev baseline, not a SECOND-finetuned model.
It consumes the shared `CLASS_SELECTION_FILE` created by Vistar's
`eval_flux2_second_oneclass_binarymask_gen.bash`: every `sample + direction`
uses exactly its recorded randomly selected target class.

FLUX Fill receives `source RGB + selected binary mask + target-class text`.
The selected mask is ground truth: **white pixels are repainted and black
pixels are preserved**. This is therefore spatially mask-conditioned, unlike
RCDGen, which receives only source RGB and target-class text.

The default short FLUX caption is `an overhead satellite image showing
<class>`. This is deliberately a content description, not a long synthetic
prompt: Fill has the mask for spatial control, but it still needs text saying
what visual content to fill. The old `change in <class>` wording is available
only as `PROMPT_MODE=legacy_change` to reproduce previous results; do not mix
its output directory with the default run.

Install a CUDA-matched PyTorch build first, then:

```bash
pip install -r requirements-flux1-fill.txt
# Accept https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev first.
huggingface-cli login
bash scripts/bootstrap_flux1_fill.sh
python tools/check_flux1_fill_deps.py
```

Run both SECOND directions with the exact Vistar record:

```bash
SECOND_ROOT=/root/data/second_dataset \
CLASS_SELECTION_FILE=/root/data/experiment/protocols/second_test_oneclass_targetmask_both_resize256_seed42_labelpairauto.jsonl \
bash run_bash/flux1_fill_second_gen.bash
```

The default is `512 -> 256`, 50 denoising steps, guidance 30, BF16, and CPU
model offload. `FLUX.1-Fill-dev` is large; use `VAE_TILING=1` when VRAM is
tight. The wrapper writes the common Vistar result layout and records that the
selected ground-truth binary mask was passed to the model.

For a fast, inspectable editing check, run several non-empty masks and save the
exact `512x512` source/mask supplied to the pipeline:

```bash
MAX_SAMPLES=8 ONLY_CHANGED=1 SAVE_MODEL_INPUTS=1 OVERWRITE=1 \
bash run_bash/flux1_fill_second_gen.bash
```

Inspect `model_input_mask/*_white_repaint_black_preserve.png` and the
per-sample `mask_area_ratio_at_model_resolution` in `prompts.jsonl`. The
default result directory includes `promptfill_target`, so the corrected run
cannot silently reuse a previous `legacy_change` prediction.
