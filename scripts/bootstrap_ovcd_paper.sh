#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
method="${METHOD:-opendpr}"
python_bin="${PYTHON_BIN:-python}"
clone_pinned() {
  local repo="$1" destination="$2" revision="$3"
  if [[ ! -d "$destination/.git" ]]; then
    if [[ -e "$destination" ]]; then echo "Existing non-Git source must be checked manually: $destination" >&2; exit 2; fi
    git clone --depth 1 "$repo" "$destination"
  fi
  if ! git -C "$destination" cat-file -e "$revision^{commit}" 2>/dev/null; then
    git -C "$destination" fetch --depth 1 origin "$revision"
  fi
  # Ordinary checkout preserves existing edits; no forced reset or clean.
  git -C "$destination" checkout --detach "$revision"
}
case "$method" in
  opendpr)
    root="${OPENDPR_ROOT:-third_party/OpenDPR}"
    clone_pinned https://github.com/guoqi2002/OpenDPR.git "$root" bdf12a3d3e5adccd4e2c166a0c3a5d9f5a19c313
    "$python_bin" -m pip install --no-build-isolation -r requirements-opendpr.txt
    "$python_bin" -m pip install --no-build-isolation -e "$root/third_party/segment_anything"
    ;;
  dynamicearth)
    "$python_bin" -m pip install --no-build-isolation -r requirements-dynamicearth.txt 'huggingface_hub<1'
    PYTHON_BIN="$python_bin" bash scripts/bootstrap_dynamicearth.sh
    clone_pinned https://github.com/facebookresearch/dino.git "${DINO_REPO:-third_party/DINO}" 7c446df5b9f45747937fb0d72314eb9f7b66930a
    ;;
  *) echo "METHOD must be opendpr or dynamicearth" >&2; exit 2 ;;
esac
"$python_bin" tools/prepare_ovcd_weights.py --method "$method"
echo "Bootstrap completed; run the import check and two-sample CUDA inference before the full queue."
