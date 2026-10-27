#!/usr/bin/env bash
# Download a SOLIDER pre-trained Swin (self-supervised on LUPerson) from the official Google Drive
# links in https://github.com/tinyvision/SOLIDER (README, "Models"), then check that it loads.
# Run from the project folder:
#
#   bash tools/download_solider.sh [DEST_DIR] [tiny|small|base]     # default: weights small
#
# Needs gdown (pip install gdown). If Google Drive refuses the download (quota), open the link
# in a browser, save the file as DEST_DIR/solider_swin_<size>.pth and run this script again.
set -euo pipefail

DEST="${1:-weights}"
SIZE="${2:-small}"
case "$SIZE" in
  tiny)  ID=12UyPVFmjoMVpQLHN07tNh4liHUmyDqg8 ;;
  small) ID=1oyEgASqDHc7YUPsQUMxuo2kBZyi2Tzfv ;;
  base)  ID=1uh7tO34tMf73MJfFqyFEGx42UBktTbZU ;;
  *) echo "size must be tiny, small or base" >&2; exit 1 ;;
esac
OUT="$DEST/solider_swin_$SIZE.pth"

mkdir -p "$DEST"
if [[ -s "$OUT" ]]; then
  echo "SOLIDER swin_$SIZE already present: $OUT"
else
  command -v gdown >/dev/null || { echo "gdown not found: pip install gdown" >&2; exit 1; }
  echo "browser link: https://drive.google.com/file/d/$ID/view"
  gdown "https://drive.google.com/uc?id=$ID" -O "$OUT.part"
  mv "$OUT.part" "$OUT"
fi

# the file must load into the Swin used by the model (prints the number of tensors loaded)
python3 -c "import sys; sys.path.insert(0, 'Scripts'); from swin import build_swin; \
build_swin('swin_$SIZE', 0.6, sys.argv[1])" "$OUT"
echo "MODEL.BACKBONE.WEIGHTS: $OUT"
