# ChangeBridge

ChangeBridge is a Table 4 candidate using source RGB plus a semantic change map.
No official task checkpoint is public, so this integration provides dataset
conversion, YAML patching, training, sampling, and output collection rather than
claiming zero-shot readiness.

The current public GitHub checkout is also missing the imported `runners/`
package. `bootstrap_changebridge.sh` and `check_changebridge_deps.py` fail loudly
on that upstream omission. Once the authors provide it, the prepared chain is:

```bash
pip install -r requirements-changebridge.txt
VISTAR_EVAL_DIR=/path/to/train/eval SPLIT=train bash run_bash/changebridge_second_prepare.bash
VISTAR_EVAL_DIR=/path/to/val/eval SPLIT=val bash run_bash/changebridge_second_prepare.bash
VISTAR_EVAL_DIR=/path/to/test/eval SPLIT=test bash run_bash/changebridge_second_prepare.bash
bash run_bash/changebridge_second_train.bash
bash run_bash/changebridge_second_sample.bash
```

Set `CHANGEBRIDGE_VQGAN_CKPT` and `CHANGEBRIDGE_CLIP_CKPT` to the official
VQGAN and SkyCLIP initializations. Sampling is testing mode (no `--train`);
the README's historical `--sample` flag is not accepted by current `main.py`.
