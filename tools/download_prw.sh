#!/usr/bin/env bash
# Download PRW from Kaggle (edoardomerli/prw-person-re-identification-in-the-wild).
# Needs the Kaggle CLI >= 1.8 (pip install kaggle) and an API token: kaggle.com -> Settings ->
# API -> "Generate New Token", saved in ~/.kaggle/access_token (or $KAGGLE_API_TOKEN).
# Legacy alternatives: ~/.kaggle/kaggle.json, or KAGGLE_USERNAME / KAGGLE_KEY.
# Without any token: download the zip in a browser, unzip it into DEST, and this script skips.
#
#   bash tools/download_prw.sh [DEST_DIR]        # default: data/PRW
#
# Does nothing if DEST_DIR already contains the dataset. Prints the folder to pass as
# --data-root / PRW_ROOT (the code also finds the dataset in a sub-folder by itself).
set -euo pipefail

DEST="${1:-data/PRW}"
SLUG="edoardomerli/prw-person-re-identification-in-the-wild"

find_root() {  # first folder below $1 that holds frame_test.mat + query_info.txt
  find "$1" -maxdepth 4 -name frame_test.mat -printf '%h\n' 2>/dev/null | while read -r d; do
    [[ -f "$d/query_info.txt" && -d "$d/annotations" && -d "$d/frames" ]] && { echo "$d"; break; }
  done
}

if [[ -d "$DEST" ]] && root="$(find_root "$DEST")" && [[ -n "$root" ]]; then
  echo "PRW already present: $root"
  exit 0
fi

command -v kaggle >/dev/null || { echo "kaggle CLI not found: pip install kaggle" >&2; exit 1; }
mkdir -p "$DEST"
kaggle datasets download -d "$SLUG" -p "$DEST" --unzip

root="$(find_root "$DEST")"
[[ -n "$root" ]] || { echo "download finished but no PRW layout found under $DEST" >&2; exit 1; }
echo "frames: $(find "$root/frames" -name '*.jpg' | wc -l)  annotations: $(find "$root/annotations" -name '*.mat' | wc -l)"
echo "PRW root: $root"
echo "export PRW_ROOT=$root"
