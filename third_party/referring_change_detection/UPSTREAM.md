# Vendored RCDGen Inference Source

This directory contains the official custom RCDGen Diffusers pipeline from:

- Repository: https://github.com/yilmazkorkmaz1/referring_change_detection
- Upstream path: `RCDGen/RCDGenSDPipeline.py`
- Upstream branch at import: `main`
- Imported: 2026-07-11

The source is vendored because RCDGen requires a custom four-channel
InstructPix2Pix pipeline that is not included in stock Diffusers 0.31.0. Model
weights are intentionally excluded and are downloaded from
`yilmazkorkmaz/RCDGen` by `scripts/bootstrap_rcdgen.sh`.
