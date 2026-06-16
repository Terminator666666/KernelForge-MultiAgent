#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${1:-$ROOT_DIR/data/flashinfer-trace}"

mkdir -p "$(dirname "$TARGET_DIR")"

if [[ -d "$TARGET_DIR" ]]; then
  echo "Dataset already exists at: $TARGET_DIR"
  exit 0
fi

if [[ -n "${FLASHINFER_TRACE_SOURCE:-}" ]]; then
  echo "Copy or extract the dataset from: $FLASHINFER_TRACE_SOURCE"
  echo "Target path: $TARGET_DIR"
  exit 0
fi

cat <<EOF
This release does not vendor the flashinfer-trace dataset.
Set FLASHINFER_TRACE_SOURCE to a local archive or copy the dataset to:
  $TARGET_DIR

Example:
  export FLASHINFER_TRACE_SOURCE=/path/to/flashinfer-trace
  ./scripts/download_data.sh
EOF
exit 1
