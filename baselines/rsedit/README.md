# RSEdit

RSEdit is a Table 4 candidate. This wrapper uses the public UNet text-ablation
checkpoint through Diffusers' stock InstructPix2Pix pipeline. The model receives
only the source image and a text instruction derived from all changed classes;
the ground-truth change mask is retained for evaluation and is never passed to
the model.

```bash
pip install -r requirements-rsedit.txt
bash scripts/bootstrap_rsedit.sh
python tools/check_rsedit_deps.py
VISTAR_EVAL_DIR=/path/to/vistar_second_eval MAX_SAMPLES=5 bash run_bash/rsedit_second_gen.bash
```

The default variant is
`BiliSakura/RSEdit-UNet-text-ablation/DGTRS-CLIP-ViT-L-14`; the bootstrap
downloads the repository snapshot and selects that subfolder. The larger
DiT release uses a separate custom pipeline and is intentionally not silently
substituted for this reproducible UNet route.
