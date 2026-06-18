#!/usr/bin/env bash
# Download official consistency-model checkpoints (ImageNet-64) to serve as the
# FROZEN ControlNet backbone. Each file is ~1.2 GB.
#
#   bash scripts/download_checkpoints.sh [DEST_DIR]
#
# By default grabs cd_imagenet64_lpips.pt (best single-step FID). For a
# CT-matched backbone, also fetch ct_imagenet64.pt (uncomment below).
set -e

DEST=${1:-checkpoints}
BASE=https://openaipublic.blob.core.windows.net/consistency
mkdir -p "$DEST"

FILES=(
  "cd_imagenet64_lpips.pt"
  # "ct_imagenet64.pt"
  # "cd_imagenet64_l2.pt"
)

dl() { command -v wget >/dev/null && wget -c -O "$2" "$1" || curl -L -C - -o "$2" "$1"; }

for f in "${FILES[@]}"; do
  echo ">> downloading $f"
  dl "$BASE/$f" "$DEST/$f"
done
echo "saved checkpoint(s) to $DEST/"
