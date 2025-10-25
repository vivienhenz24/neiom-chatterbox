#!/usr/bin/env bash
# Automate multilingual TTS fine-tuning end-to-end: dependency install, model/data downloads,
# token preparation, config generation, and training. Designed to run right after cloning.

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/run_full_finetune.sh [options]

Options:
  --python PATH            Python interpreter to use (default: python3)
  --device DEVICE          Training device passed to train_t3 (default: cuda)
  --prep-device DEVICE     Device for data preparation tokenizer (default: cpu)
  --tokens-dir PATH        Directory for prepared token files (default: data/luxembourgish_tokens)
  --output-dir PATH        Logs/checkpoints output directory (default: runs/runpod_luxembourgish)
  --config PATH            Destination of the generated fine-tuning config (default: configs/runpod_luxembourgish.yaml)
  --train-split NAME       Dataset split used for training (default: train)
  --valid-split NAME       Dataset split used for validation (default: test; pass "none" to disable)
  --max-token-len N        Optional speech token truncation length during preparation
  --resume PATH            Resume training from checkpoint PATH
  --skip-install           Skip dependency installation
  --skip-model             Skip multilingual model download
  --skip-data              Skip Luxembourgish corpus download/validation
  --skip-prepare           Skip token preparation (expects tokens already exist)
  --skip-train             Skip the training run
  --keep-config            Do not overwrite the config file if it already exists
  --no-skip-existing       Recompute token files even if they are already present
  --eval-only              Run evaluation only (passes --eval-only to train_t3)
  --no-validation          Disable validation during training (--no-validation)
  --amp                    Force AMP even if config disables it (--amp)
  --help                   Show this message and exit

Environment:
  HF_TOKEN must be exported to access Hugging Face-hosted corpora and checkpoints.
USAGE
}

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
}

fail() {
  log "ERROR: $*"
  exit 1
}

ensure_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    fail "Required command '$1' not found. Install it or adjust PATH."
  fi
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAIN_DEVICE="cuda"
PREP_DEVICE="cpu"
TOKENS_ROOT="data/luxembourgish_tokens"
OUTPUT_DIR="runs/runpod_luxembourgish"
CONFIG_PATH="configs/runpod_luxembourgish.yaml"
MODEL_DIR="models/multilingual"
TRAIN_SPLIT="train"
VALID_SPLIT="test"
MAX_TOKEN_LEN=""
RESUME_PATH=""
INSTALL_DEPS=1
RUN_DOWNLOAD_MODEL=1
RUN_DOWNLOAD_DATA=1
RUN_PREPARE=1
RUN_TRAIN=1
KEEP_CONFIG=0
SKIP_EXISTING=1
EVAL_ONLY=0
NO_VALIDATION=0
FORCE_AMP=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_BIN="$2"; shift 2 ;;
    --device)
      TRAIN_DEVICE="$2"; shift 2 ;;
    --prep-device)
      PREP_DEVICE="$2"; shift 2 ;;
    --tokens-dir)
      TOKENS_ROOT="$2"; shift 2 ;;
    --output-dir)
      OUTPUT_DIR="$2"; shift 2 ;;
    --config)
      CONFIG_PATH="$2"; shift 2 ;;
    --train-split)
      TRAIN_SPLIT="$2"; shift 2 ;;
    --valid-split)
      VALID_SPLIT="$2"; shift 2 ;;
    --max-token-len)
      MAX_TOKEN_LEN="$2"; shift 2 ;;
    --resume)
      RESUME_PATH="$2"; shift 2 ;;
    --skip-install)
      INSTALL_DEPS=0; shift ;;
    --skip-model)
      RUN_DOWNLOAD_MODEL=0; shift ;;
    --skip-data)
      RUN_DOWNLOAD_DATA=0; shift ;;
    --skip-prepare)
      RUN_PREPARE=0; shift ;;
    --skip-train)
      RUN_TRAIN=0; shift ;;
    --keep-config)
      KEEP_CONFIG=1; shift ;;
    --no-skip-existing)
      SKIP_EXISTING=0; shift ;;
    --eval-only)
      EVAL_ONLY=1; shift ;;
    --no-validation)
      NO_VALIDATION=1; shift ;;
    --amp)
      FORCE_AMP=1; shift ;;
    --help)
      usage; exit 0 ;;
    *)
      usage; fail "Unknown option: $1" ;;
  esac
done

ensure_cmd "$PYTHON_BIN"

python_version_check=$("$PYTHON_BIN" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)

if [[ "$python_version_check" == 3.12* || "$python_version_check" == 3.13* ]]; then
  if command -v python3.11 >/dev/null 2>&1; then
    log "Switching to python3.11 to preserve compatibility with dependency pins."
    PYTHON_BIN="python3.11"
  elif command -v apt-get >/dev/null 2>&1; then
    log "Installing python3.11 for compatibility..."
    DEBIAN_FRONTEND=noninteractive apt-get update -y
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3.11 python3.11-venv
    PYTHON_BIN="python3.11"
  else
    fail "Python ${python_version_check} detected but python3.11 unavailable. Install Python 3.11 or pass --python PATH."
  fi
  ensure_cmd "$PYTHON_BIN"
fi

resolve_path() {
  "$PYTHON_BIN" -c 'import pathlib, sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve())' "$1"
}

if [[ ( "$RUN_DOWNLOAD_MODEL" -eq 1 || "$RUN_DOWNLOAD_DATA" -eq 1 ) && -z "${HF_TOKEN:-}" ]]; then
  fail "HF_TOKEN is not set. Export your Hugging Face token before running."
fi

TOKENS_ROOT_ABS="$(resolve_path "$TOKENS_ROOT")"
OUTPUT_DIR_ABS="$(resolve_path "$OUTPUT_DIR")"
CONFIG_PATH_ABS="$(resolve_path "$CONFIG_PATH")"
MODEL_DEST_ABS="$(resolve_path "$MODEL_DIR")"

mkdir -p "$TOKENS_ROOT_ABS" "$OUTPUT_DIR_ABS" "$(dirname "$CONFIG_PATH_ABS")" "$MODEL_DEST_ABS"

if (( INSTALL_DEPS )); then
  log "Installing project and dataset requirements..."
  if command -v apt-get >/dev/null 2>&1; then
    log "Ensuring system audio dependencies via apt-get..."
    DEBIAN_FRONTEND=noninteractive apt-get update -y
    DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg libsndfile1
  fi

  "$PYTHON_BIN" -m pip install --upgrade pip
  "$PYTHON_BIN" -m pip install -e "$REPO_ROOT"
  "$PYTHON_BIN" -m pip install -r "$REPO_ROOT/requirements-luxembourgish.txt"

  if ! "$PYTHON_BIN" - <<'PY'
import sys, torch
sys.exit(0 if torch.cuda.is_available() else 1)
PY
  then
    log "Reinstalling torch/torchaudio with CUDA-enabled wheels..."
    "$PYTHON_BIN" -m pip install --upgrade torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
    if ! "$PYTHON_BIN" - <<'PY'
import sys, torch
sys.exit(0 if torch.cuda.is_available() else 1)
PY
    then
      fail "Torch still reports no CUDA support after reinstall. Verify the pod image includes NVIDIA drivers."
    fi
  fi

  if [[ "${HF_HUB_ENABLE_HF_TRANSFER:-}" =~ ^([Tt][Rr][Uu][Ee]|1|yes|YES)$ ]]; then
    if ! "$PYTHON_BIN" -m pip show hf_transfer >/dev/null 2>&1; then
      log "Installing hf_transfer to satisfy HF_HUB_ENABLE_HF_TRANSFER..."
      "$PYTHON_BIN" -m pip install hf_transfer
    fi
  fi
fi

if [[ "$PREP_DEVICE" == cuda* ]]; then
  if ! "$PYTHON_BIN" - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    sys.exit(1)
major, _ = torch.cuda.get_device_capability(0)
sys.exit(0 if major <= 9 else 2)
PY
  then
    status=$?
    if [[ $status -eq 1 ]]; then
      log "CUDA unavailable; falling back to CPU for data preparation."
    elif [[ $status -eq 2 ]]; then
      log "CUDA device capability not supported by this PyTorch build; falling back to CPU for data preparation."
    else
      log "Unable to validate CUDA capability; falling back to CPU for data preparation."
    fi
    PREP_DEVICE="cpu"
  fi
fi

if (( RUN_DOWNLOAD_MODEL )); then
  log "Downloading multilingual base checkpoint into $MODEL_DEST_ABS ..."
  "$PYTHON_BIN" "$REPO_ROOT/download_multilingual_model.py" --dest "$MODEL_DEST_ABS"
fi

DATA_BUILD_SCRIPT="$REPO_ROOT/data/scripts/build_and_validate_luxembourgish.sh"
if (( RUN_DOWNLOAD_DATA )); then
  [[ -x "$DATA_BUILD_SCRIPT" ]] || fail "Dataset build script missing at $DATA_BUILD_SCRIPT"
  log "Downloading and validating Luxembourgish corpus..."
  HF_TOKEN="${HF_TOKEN:-}" PYTHON_BIN="$PYTHON_BIN" bash "$DATA_BUILD_SCRIPT"
fi

DATA_ROOT="$REPO_ROOT/data/luxembourgish_corpus"
COMBINED_ROOT="$DATA_ROOT/luxembourgish_combined"
DATASET_BASE="$COMBINED_ROOT"
if [[ ! -d "$COMBINED_ROOT" ]]; then
  DATASET_BASE="$DATA_ROOT"
fi

if [[ ! -d "$DATASET_BASE" ]]; then
  fail "Combined Luxembourgish dataset not found under $DATA_ROOT. Check the download step above for errors."
fi

if [[ "$DATASET_BASE" == "$DATA_ROOT" ]]; then
  log "Using dataset root $DATA_ROOT (no 'luxembourgish_combined' subdirectory present)."
else
  log "Using dataset root $COMBINED_ROOT."
fi

TRAIN_SPLIT_DIR="$DATASET_BASE/$TRAIN_SPLIT"
VALID_SPLIT_DIR=""
if [[ "$VALID_SPLIT" != "none" ]]; then
  VALID_SPLIT_DIR="$DATASET_BASE/$VALID_SPLIT"
fi

TOKENIZER_JSON="$MODEL_DEST_ABS/grapheme_mtl_merged_expanded_v1.json"
if (( RUN_PREPARE )); then
  [[ -f "$TOKENIZER_JSON" ]] || fail "Tokenizer JSON not found (expected $TOKENIZER_JSON). Run model download first."
  [[ -d "$TRAIN_SPLIT_DIR" ]] || fail "Training split directory not found: $TRAIN_SPLIT_DIR"

  prepare_split() {
    local split_dir="$1"
    local split_name="$2"
    local split_dir_abs
    split_dir_abs="$(resolve_path "$split_dir")"
    local target_dir_abs
    target_dir_abs="$(resolve_path "$TOKENS_ROOT_ABS/$split_name")"
    mkdir -p "$target_dir_abs"

    [[ -d "$split_dir_abs" ]] || fail "Dataset split directory missing: $split_dir_abs"

    log "Preparing tokens for split '$split_name' → $target_dir_abs"
    local args=(
      -m chatterbox.training.data_preparation
      --dataset-root "$split_dir_abs"
      --metadata metadata.csv
      --output-dir "$target_dir_abs"
      --tokenizer-path "$TOKENIZER_JSON"
      --language-id lb
      --device "$PREP_DEVICE"
    )
    if [[ -n "$MAX_TOKEN_LEN" ]]; then
      args+=(--max-token-len "$MAX_TOKEN_LEN")
    fi
    if (( SKIP_EXISTING )); then
      args+=(--skip-existing)
    fi
    "$PYTHON_BIN" "${args[@]}"
  }

  prepare_split "$TRAIN_SPLIT_DIR" "$TRAIN_SPLIT"
  if [[ -n "$VALID_SPLIT_DIR" ]]; then
    prepare_split "$VALID_SPLIT_DIR" "$VALID_SPLIT"
  fi
fi

TRAIN_TOKENS_DIR_ABS="$(resolve_path "$TOKENS_ROOT_ABS/$TRAIN_SPLIT")"
[[ -d "$TRAIN_TOKENS_DIR_ABS" ]] || fail "Train tokens directory missing: $TRAIN_TOKENS_DIR_ABS"

if ! find "$TRAIN_TOKENS_DIR_ABS" -maxdepth 1 -name '*.pt' -print -quit >/dev/null; then
  fail "No token files were generated in $TRAIN_TOKENS_DIR_ABS. Check the data preparation logs above for missing audio/text warnings."
fi

VALID_TOKENS_DIR_ABS=""
if [[ -n "$VALID_SPLIT_DIR" ]]; then
  VALID_TOKENS_DIR_ABS="$(resolve_path "$TOKENS_ROOT_ABS/$VALID_SPLIT")"
  [[ -d "$VALID_TOKENS_DIR_ABS" ]] || fail "Validation tokens directory missing: $VALID_TOKENS_DIR_ABS"
  if ! find "$VALID_TOKENS_DIR_ABS" -maxdepth 1 -name '*.pt' -print -quit >/dev/null; then
    fail "No token files were generated in $VALID_TOKENS_DIR_ABS. Check the data preparation logs above for missing audio/text warnings."
  fi
fi

if (( KEEP_CONFIG )); then
  [[ -f "$CONFIG_PATH_ABS" ]] || fail "--keep-config set but config file not found at $CONFIG_PATH_ABS"
  log "Keeping existing training config at $CONFIG_PATH_ABS"
else
  BATCH_SIZE=4
  GRAD_ACCUM=4
  EPOCHS=2
  LR="1.0e-5"
  WEIGHT_DECAY="0.01"

  VALID_ARG="NONE"
  if [[ -n "$VALID_TOKENS_DIR_ABS" ]]; then
    VALID_ARG="$VALID_TOKENS_DIR_ABS"
  fi

  mapfile -t STATS < <("$PYTHON_BIN" - "$TRAIN_TOKENS_DIR_ABS" "$VALID_ARG" "$BATCH_SIZE" "$GRAD_ACCUM" "$EPOCHS" <<'PY'
import math
import sys
from pathlib import Path
from chatterbox.training.datasets.t3_dataset import T3TokenDataset

train_dir = Path(sys.argv[1])
valid_arg = sys.argv[2]
batch = int(sys.argv[3])
grad = int(sys.argv[4])
epochs = int(sys.argv[5])

train_ds = T3TokenDataset(train_dir, drop_missing_text=True)
effective_batch = batch * grad
steps_per_epoch = max(1, math.ceil(len(train_ds) / effective_batch))
total_steps = max(steps_per_epoch * epochs, 1)
valid_len = 0
if valid_arg != "NONE":
    valid_dir = Path(valid_arg)
    if valid_dir.exists():
        valid_len = len(T3TokenDataset(valid_dir, drop_missing_text=True))

print(len(train_ds))
print(valid_len)
print(steps_per_epoch)
print(total_steps)
print(train_ds.stats.max_speech_tokens)
PY
)

  TRAIN_SAMPLES="${STATS[0]}"
  VALID_SAMPLES="${STATS[1]}"
  STEPS_PER_EPOCH="${STATS[2]}"
  TOTAL_STEPS="${STATS[3]}"
  MAX_SPEECH_TOKENS="${STATS[4]}"

  log "Dataset stats → train: ${TRAIN_SAMPLES} samples, valid: ${VALID_SAMPLES} samples, steps/epoch: ${STEPS_PER_EPOCH}, max speech tokens: ${MAX_SPEECH_TOKENS}"

  WARMUP_STEPS=500
  if [[ "$TOTAL_STEPS" -lt 500 ]]; then
    WARMUP_STEPS=$(( TOTAL_STEPS / 5 ))
    if [[ "$WARMUP_STEPS" -lt 10 ]]; then
      WARMUP_STEPS=10
    fi
    if [[ "$WARMUP_STEPS" -ge "$TOTAL_STEPS" ]]; then
      HALF=$(( TOTAL_STEPS / 2 ))
      if [[ "$HALF" -lt 1 ]]; then
        HALF=1
      fi
      WARMUP_STEPS=$HALF
    fi
  fi

  EVAL_EVERY=$STEPS_PER_EPOCH
  if [[ "$EVAL_EVERY" -gt 250 ]]; then
    EVAL_EVERY=250
  fi
  if [[ "$EVAL_EVERY" -lt 1 ]]; then
    EVAL_EVERY=1
  fi

  CHECKPOINT_EVERY=$EVAL_EVERY
  if [[ "$CHECKPOINT_EVERY" -lt 100 ]]; then
    CHECKPOINT_EVERY=$(( EVAL_EVERY * 2 ))
  fi
  if [[ "$CHECKPOINT_EVERY" -lt "$EVAL_EVERY" ]]; then
    CHECKPOINT_EVERY=$EVAL_EVERY
  fi
  if [[ "$CHECKPOINT_EVERY" -gt 500 ]]; then
    CHECKPOINT_EVERY=500
  fi

  VALID_YAML_VALUE="null"
  if [[ -n "$VALID_TOKENS_DIR_ABS" ]]; then
    VALID_YAML_VALUE="\"$VALID_TOKENS_DIR_ABS\""
  fi

  cat >"$CONFIG_PATH_ABS" <<EOF
dataset:
  train_tokens_dir: "${TRAIN_TOKENS_DIR_ABS}"
  valid_tokens_dir: ${VALID_YAML_VALUE}
  batch_size: ${BATCH_SIZE}
  eval_batch_size: ${BATCH_SIZE}
  num_workers: 4
  max_source_tokens: null
  max_target_tokens: null

model:
  base_checkpoint: "${MODEL_DEST_ABS}/t3_mtl23ls_v2.safetensors"
  freeze_encoder: true
  freeze_decoder: false
  freeze_modules: []

optimizer:
  name: adamw
  lr: ${LR}
  betas: [0.9, 0.999]
  eps: 1.0e-8
  weight_decay: ${WEIGHT_DECAY}

scheduler:
  name: linear
  warmup_steps: ${WARMUP_STEPS}
  min_lr: 1.0e-6

training:
  epochs: ${EPOCHS}
  gradient_accumulation_steps: ${GRAD_ACCUM}
  mixed_precision: true
  max_grad_norm: 1.0
  eval_every_n_steps: ${EVAL_EVERY}

logging:
  output_dir: "${OUTPUT_DIR_ABS}"
  log_every_n_steps: 50
  tensorboard_enabled: true
  wandb_enabled: false
  wandb_project: null
  wandb_run_name: null
  checkpoint_every_n_steps: ${CHECKPOINT_EVERY}
  max_checkpoints: 3

seed:
  python: 1337
  numpy: null
  torch: null
EOF

  log "Generated training config at ${CONFIG_PATH_ABS} (warmup_steps=${WARMUP_STEPS}, eval_every=${EVAL_EVERY})."
fi

log "Using training config at $CONFIG_PATH_ABS"

if (( RUN_TRAIN )); then
  local_log="$OUTPUT_DIR_ABS/train.log"
  log "Launching fine-tuning; logging to $local_log"

  cmd=("$PYTHON_BIN" "-m" "chatterbox.training.train_t3" "$CONFIG_PATH_ABS" "--device" "$TRAIN_DEVICE" "--log-file" "$local_log")
  if [[ -n "$RESUME_PATH" ]]; then
    cmd+=("--resume" "$RESUME_PATH")
  fi
  if (( EVAL_ONLY )); then
    cmd+=("--eval-only")
  fi
  if (( NO_VALIDATION )); then
    cmd+=("--no-validation")
  fi
  if (( FORCE_AMP )); then
    cmd+=("--amp")
  fi
  "${cmd[@]}"
fi

log "Pipeline complete."
