#!/usr/bin/env python3
"""
Utility script to download the Chatterbox multilingual TTS model
artifacts from Hugging Face into a local directory inside this repo.
Requires an `HF_TOKEN` entry in `.env` or the environment.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict

from huggingface_hub import snapshot_download


REPO_ID = "ResembleAI/chatterbox"
DEFAULT_ALLOW_PATTERNS = [
    "ve.pt",
    "t3_mtl23ls_v2.safetensors",
    "s3gen.pt",
    "grapheme_mtl_merged_expanded_v1.json",
    "conds.pt",
    "Cangjie5_TC.json",
]


def load_env_file(env_path: Path) -> Dict[str, str]:
    """Basic .env loader that returns key/value pairs."""
    if not env_path.exists():
        return {}

    loaded: Dict[str, str] = {}
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        loaded[key.strip()] = value.strip().strip('"').strip("'")
    return loaded


def ensure_hf_token(env_path: Path) -> str:
    """Return an HF token, preferring environment, falling back to .env."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token

    env_values = load_env_file(env_path)
    token = env_values.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is not set. Add it to the environment or `.env` file."
        )

    os.environ["HF_TOKEN"] = token
    return token


def download_model(destination: Path, revision: str) -> Path:
    """Download multilingual model weights into `destination`."""
    destination.mkdir(parents=True, exist_ok=True)
    local_dir = snapshot_download(
        repo_id=REPO_ID,
        repo_type="model",
        revision=revision,
        allow_patterns=DEFAULT_ALLOW_PATTERNS,
        local_dir=str(destination),
        local_dir_use_symlinks=False,
        token=os.environ.get("HF_TOKEN"),
    )
    return Path(local_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the Chatterbox multilingual model locally."
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("models/multilingual"),
        help="Directory to save the multilingual model files (default: models/multilingual).",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Hugging Face repo revision to download (default: main).",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Path to the .env file containing HF_TOKEN (default: .env).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        token = ensure_hf_token(args.env_file)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Using HF token from {args.env_file if token else 'environment'}")
    print(f"Downloading multilingual model from {REPO_ID}@{args.revision}...")
    target_dir = download_model(args.dest, args.revision)
    print(f"Multilingual model files are available in: {target_dir.resolve()}")


if __name__ == "__main__":
    main()
