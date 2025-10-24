#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${ROOT_DIR}/luxembourgish_corpus"
COMBINED_DIR="${WORK_DIR}"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python3}}"

if [ ! -x "$(command -v "${PYTHON_BIN}")" ]; then
  echo "[ERROR] Python interpreter not found: ${PYTHON_BIN}" >&2
  exit 1
fi

ENV_FILE="${ROOT_DIR}/.env"
if [ -f "${ENV_FILE}" ]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

if [ -z "${HF_TOKEN:-}" ]; then
  echo "[WARN] HF_TOKEN is not set. Downloads from the Hugging Face Hub may fail."
fi

# Remove legacy combined folder if it exists
if [ -d "${WORK_DIR}/luxembourgish_combined" ]; then
  rm -rf "${WORK_DIR}/luxembourgish_combined"
fi

echo "[INFO] Preparing combined corpus..."
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/prepare_luxembourgish_combined.py" --work-dir "${WORK_DIR}"

echo "[INFO] Validating combined corpus..."
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/check_luxembourgish_combined.py" \
  --dataset-dir "${COMBINED_DIR}" \
  --work-dir "${WORK_DIR}"

echo "[INFO] Done."
