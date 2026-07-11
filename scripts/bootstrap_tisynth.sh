#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TISYNTH_ROOT="${TISYNTH_ROOT:-${ROOT_DIR}/third_party/TISynth}"
EXPECTED_REVISION="688cda0597b1550cb32bbae0469b8a0a900501c0"

if [[ ! -f "${TISYNTH_ROOT}/batch_infer.py" ]]; then
  echo "[bootstrap_tisynth] Missing vendored TISynth source: ${TISYNTH_ROOT}" >&2
  echo "This repository must include third_party/TISynth; no network download is performed." >&2
  exit 1
fi
if [[ ! -f "${TISYNTH_ROOT}/UPSTREAM.md" ]]; then
  echo "[bootstrap_tisynth] Missing vendoring metadata: ${TISYNTH_ROOT}/UPSTREAM.md" >&2
  exit 1
fi

echo "[bootstrap_tisynth] using vendored official source: ${TISYNTH_ROOT}"
echo "[bootstrap_tisynth] pinned upstream revision: ${EXPECTED_REVISION}"
echo "[bootstrap_tisynth] zero-shot protocol: use the official GID_model.ckpt directly on LoveDA."
