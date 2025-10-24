#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${ROOT_DIR}/luxembourgish_corpus"
COMBINED_DIR="${WORK_DIR}/luxembourgish_combined"

echo "[INFO] Preparing combined corpus..."
python "${ROOT_DIR}/scripts/prepare_luxembourgish_combined.py" --work-dir "${WORK_DIR}"

echo "[INFO] Validating combined corpus..."
python "${ROOT_DIR}/scripts/check_luxembourgish_combined.py" \
  --dataset-dir "${COMBINED_DIR}" \
  --work-dir "${WORK_DIR}"

echo "[INFO] Done."
