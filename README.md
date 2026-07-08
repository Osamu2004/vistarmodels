# vistarmodels

Baselines and wrappers for evaluating generation models against Vistar/UniGen.

This repository is intentionally separate from the main `vistar` codebase. It
keeps external comparison methods reproducible without mixing their dependencies
and source code into the training repository.

## Baselines

- `baselines/crsdiff`: CRS-Diff wrapper for remote-sensing controllable image
  generation.

