# SECOND semantic-synthesis baselines

SPADE, OASIS, and ControlNet are trained in-domain on the common bidirectional
SECOND training set prepared by `paper_baselines_prepare_data.bash`. Their input
is the target-side directional semantic mask only; source imagery is not passed
to these three models. The test set has 3388 directional items at 256x256.

SPADE and OASIS use their official repositories. The maintained SD 1.5
ControlNet path uses the Diffusers model components through a direct SECOND
manifest trainer, with class-aware text in addition to the target-side mask.
Its default training configuration is one GPU with batch size 2. See
`baselines/controlnet/README.md` for the exact train, resume, generation, and
common-evaluation commands. All model outputs are collected into the same
VISTAR evaluation folder contract before the shared Table 4 metric script is
run.
