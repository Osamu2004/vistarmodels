#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MMSeg_CONFIG="${MMSeg_CONFIG:?set MMSeg_CONFIG}"
MMSeg_CKPT="${MMSeg_CKPT:?set MMSeg_CKPT}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/data/experiment/todsynth_crfm_loveda_eval}"
export OUTPUT_DIR
bash "${ROOT_DIR}/run_bash/todsynth_loveda_gen.bash" --crfm --mmseg_config "${MMSeg_CONFIG}" --mmseg_ckpt "${MMSeg_CKPT}" "${@}"
