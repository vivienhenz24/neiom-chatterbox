"""
Convert speech waveforms into S3 speech tokens for T3 fine-tuning.

Example:
    python -m chatterbox.training.data_preparation \\
        --dataset-root data/luxembourgish_corpus/train \\
        --metadata metadata.csv \\
        --output-dir tokens
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Optional

import librosa
import torch
from tqdm import tqdm

from ..models.s3tokenizer import S3Tokenizer, S3_SR


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare S3 speech tokens from text/audio metadata.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path to the dataset split containing metadata and audio/.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("metadata.csv"),
        help="CSV file with at least an audio path column (relative to dataset root).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tokens"),
        help="Directory where token files (.pt) will be written.",
    )
    parser.add_argument(
        "--path-field",
        type=str,
        default="path",
        help="Column name in the metadata CSV containing paths to audio files.",
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default="text",
        help="Optional column name for transcriptions to store alongside tokens.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for the tokenizer (e.g., cpu, cuda).",
    )
    parser.add_argument(
        "--max-token-len",
        type=int,
        default=None,
        help="Truncate output speech tokens to at most this length.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not recompute samples whose token file already exists.",
    )
    return parser.parse_args()


def load_audio(path: Path) -> torch.Tensor:
    wav, _ = librosa.load(path, sr=S3_SR)
    return torch.from_numpy(wav).float()


def tokenize_audio(
    tokenizer: S3Tokenizer,
    audio: torch.Tensor,
    max_len: Optional[int] = None,
) -> tuple[torch.Tensor, int]:
    tokens, lengths = tokenizer.forward([audio], max_len=max_len)
    tokens = tokens.squeeze(0)
    length = int(lengths.squeeze(0))
    return tokens[:length].cpu(), length


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    dataset_root = args.dataset_root
    metadata_path = args.metadata if args.metadata.is_absolute() else dataset_root / args.metadata
    output_dir = args.output_dir if args.output_dir.is_absolute() else dataset_root / args.output_dir

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    tokenizer = S3Tokenizer()
    tokenizer.to(device)
    tokenizer.eval()

    logger.info("Loaded S3Tokenizer on %s", device)
    logger.info("Writing token files to %s", output_dir)

    with metadata_path.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        if args.path_field not in reader.fieldnames:
            raise KeyError(f"Column '{args.path_field}' not found in {metadata_path}")

        for row in tqdm(reader, desc="Tokenizing", dynamic_ncols=True):
            audio_rel = Path(row[args.path_field])
            audio_path = dataset_root / audio_rel
            if not audio_path.exists():
                logger.warning("Missing audio file: %s", audio_path)
                continue

            sample_id = audio_rel.stem
            out_path = output_dir / f"{sample_id}.pt"
            if args.skip_existing and out_path.exists():
                continue

            audio = load_audio(audio_path).to(device)
            tokens, length = tokenize_audio(tokenizer, audio, max_len=args.max_token_len)

            payload = {
                "speech_tokens": tokens,
                "speech_token_len": length,
                "audio_path": str(audio_rel),
            }

            if args.text_field and args.text_field in row:
                payload["text"] = row[args.text_field]

            torch.save(payload, out_path)

    logger.info("Done.")


if __name__ == "__main__":
    main()

