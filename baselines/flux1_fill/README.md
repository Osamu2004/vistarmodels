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

The default is `512 -> 256`, 50 denoising steps, guidance 30, BF16, and full
CUDA residency (`CPU_OFFLOAD=0`). This prevents the VAE, text encoders, and 12B
Transformer from being moved between host memory and GPU memory again for each
sample. Use `CPU_OFFLOAD=1` only when the full pipeline does not fit. An
optional `CPU_OFFLOAD=auto` mode selects full CUDA when at least 44 GiB is free.
`FLUX.1-Fill-dev` is large; use `VAE_TILING=1` when VRAM is tight. The wrapper
writes the common Vistar result layout and records that the selected
ground-truth binary mask was passed to the model.

The six class captions are encoded once and cached on the execution device.
This avoids rerunning CLIP/T5 and swapping them with the Transformer between
all 3,388 samples. `runtime_config.json` records placement and prompt-cache
startup time; each generated row in `prompts.jsonl` records generation time and
cache hits. Set `CACHE_PROMPT_EMBEDS=0` only for an exact performance ablation.
The inner 50-step bar is hidden by default so the outer bar reports real
per-sample generation time; restore it with `SHOW_DENOISING_PROGRESS=1`.

The normal fast resident command is:

```bash
CUDA_VISIBLE_DEVICES=1 CPU_OFFLOAD=0 CACHE_PROMPT_EMBEDS=1 \
bash run_bash/flux1_fill_second_gen.bash
```

If this raises CUDA OOM, rerun with `CPU_OFFLOAD=1`. Offload necessarily reloads
large components between samples, although prompt caching still avoids repeated
CLIP/T5 execution. Existing valid predictions are skipped, so changing
placement does not require deleting the output folder.

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
