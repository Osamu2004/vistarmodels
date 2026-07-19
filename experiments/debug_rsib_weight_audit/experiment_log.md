# RSIB weight and inference audit

## Problem

Determine whether GSNet may reuse the previously downloaded
`/root/data/weight/rsib/RSIB.pth`, whether the prior RSKT-Seg adapter used the
official model correctly, and why the earlier bootstrap downloaded RSIB
repeatedly.

## Evidence

- The official RSKT-Seg README and official GSNet README link exactly the same
  specialist checkpoint: Google Drive file ID
  `1kH0wDM_Hl4sEQJG8JjILCo0RTx65X7zV`.
- `RSKT_Seg/RSKT_Seg.py::BuildRSIB` and
  `gs_net/GSNet.py::BuildRSIB` are textually identical.
- Both construct `vit_base(patch_size=8, num_classes=0)`, select the
  checkpoint's `teacher` entry, remove `module.` and `backbone.` prefixes, and
  load with `strict=False`.
- The two official `vision_transformer.py` files have the same SHA-256:
  `d241d93a45a7a0aa6893d3f35ccb40bd49544382d6859fb46b73edc51416306e`.
- RSKT-Seg uses the user-selected DLRSD ViT-L/336 layer-5 main checkpoint,
  ViT-L/14@336, RemoteCLIP ViT-B/32, and RSIB. GSNet uses its independent
  LandDiscover50K main checkpoint, ViT-B/16, and the shared RSIB.
- The RSKT-Seg adapter uses the official model builder and official config,
  overrides only target vocabulary/path fields and the layer count matching
  the selected `layer5` checkpoint, and follows the official non-sliding
  `[1,1]` pooling evaluation setting.
- External native 512-by-512 non-overlapping tiling is an intentional unified
  evaluation adaptation. Each source tile is supplied at 512 without external
  resize; the official RSKT-Seg model internally resamples its CLIP and RSIB
  branches to their fixed encoder resolutions.
- Completed runs loaded the official model and produced full CHN6-CUG and
  xBD-pre metrics, which verifies the executable path after the PyTorch 2.6
  compatibility patch.

## Root cause

The earlier repeated downloads were caused by the bootstrap validator, not by
the inference graph or by a wrong RSIB file. The released RSIB checkpoint is a
legacy trusted PyTorch checkpoint containing a NumPy scalar. PyTorch 2.6
changed the default of `torch.load` to `weights_only=True`; the validator used
that restricted mode, rejected the valid file, deleted it, and retried the
download. The actual RSKT-Seg inference loader had already been corrected to
use `weights_only=False` only while loading explicitly supplied official
release weights.

## Fix

- Validate the trusted official RSIB release with `weights_only=False`.
- Limit Google Drive retries to three.
- Pin fresh RSKT-Seg clones to official revision
  `7b84091598e1edc3236dfbf45cc27e7e3436ffcb`.
- Report the checked RSKT-Seg source revision in the dependency audit.
- Install the bundled Detectron2 with `--no-build-isolation`, so its build can
  see the CUDA-enabled PyTorch already installed in the environment.
- Declare the previously implicit `matplotlib` and `cloudpickle` runtime
  dependencies.
- Keep the shared RSIB default path for both RSKT-Seg and GSNet.

## Conclusion

The prior RSKT-Seg inference implementation and the current GSNet model wiring
use the intended official architectures and weights. The faulty component was
only the earlier RSKT-Seg download validator. Reusing the single downloaded
RSIB file for both methods is the official and storage-efficient setup.
