#!/usr/bin/env bash
set -euo pipefail

batch_root="${1:-/home/depp/AIC/AIC26/batch1}"
workers="${WORKERS:-8}"
mapfile -t video_ids < <(find "$batch_root/video" -maxdepth 1 -type f -name '*.mp4' -printf '%f\n' | sed 's/\.mp4$//' | sort)

if ((${#video_ids[@]} == 0)); then
  echo "No MP4 videos found in $batch_root/video" >&2
  exit 1
fi

.venv/bin/python -m backend.offline_indexing.cli preprocess \
  "$batch_root/video" "${video_ids[@]}" \
  --output "$batch_root" \
  --skip-existing \
  --workers "$workers"
