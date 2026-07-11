# FLUX.1 Fill-dev on SECOND

This is a zero-shot FLUX.1 Fill-dev baseline, not a SECOND-finetuned model.
It consumes the shared `CLASS_SELECTION_FILE` created by Vistar's
`eval_flux2_second_oneclass_binarymask_gen.bash`: every `sample + direction`
uses exactly its recorded randomly selected target class.

FLUX Fill receives `source RGB + selected binary mask + target-class text`.
The selected mask is ground truth: white pixels may be regenerated, black
pixels must be preserved. This is therefore spatially mask-conditioned, unlike
RCDGen, which receives only source RGB and target-class text.

The text is intentionally minimal and matches the RCDGen style:
`change in <class>`. No extra scene or preservation prompt is appended.

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
