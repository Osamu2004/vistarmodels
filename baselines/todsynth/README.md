# TODSynth / CRFM

TODSynth is a Table 3 candidate. The public CVPR 2026 code provides training and
demo inference but no task checkpoint. Its `test.py` and `crfm_test.py` stop
after the first batch, and the public CRFM collate function drops the target
semantic tensor. The public inference modules also reference their pipeline
output type without importing it. `run_todsynth_manifest.py` fixes these
integration problems
without modifying upstream source: it iterates the full dataset, preserves
sample names, retains CRFM targets, and writes the common LoveDA folders.

```bash
pip install -r requirements-todsynth.txt
bash scripts/bootstrap_todsynth.sh
VISTAR_EVAL_DIR=/path/to/loveda/trainval SPLITS=train DATASET_TAG=loveda_train \
  bash run_bash/todsynth_loveda_prepare.bash
bash run_bash/todsynth_loveda_vectorize.bash
bash run_bash/todsynth_loveda_train.bash
VISTAR_EVAL_DIR=/path/to/loveda/trainval SPLITS=val DATASET_TAG=loveda_val \
  bash run_bash/todsynth_loveda_prepare.bash
DATASET_TAG=loveda_val bash run_bash/todsynth_loveda_vectorize.bash
DATASET_TAG=loveda_val TODSYNTH_CHECKPOINT=/path/to/checkpoint bash run_bash/todsynth_loveda_gen.bash
DATASET_TAG=loveda_val MMSeg_CONFIG=... MMSeg_CKPT=... TODSYNTH_CHECKPOINT=... \
  bash run_bash/todsynth_loveda_crfm_gen.bash
```

`stabilityai/stable-diffusion-3.5-medium` is gated. Store an accepted local
snapshot on fast storage and point `SD35_MODEL_DIR` to it.
CRFM additionally requires a LoveDA-trained MMSegmentation model; it is a
sampling refinement, not a substitute for the trained TODSynth checkpoint.
Prepare validation/test data into a different `DATASET_DIR` with `SPLITS=val`
or `SPLITS=test`; do not train on the evaluation split.
