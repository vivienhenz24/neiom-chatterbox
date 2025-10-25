"""
Convert speech waveforms and transcripts into S3 / text tokens for T3 fine-tuning.

Example:
    python -m chatterbox.training.data_preparation \\
        --dataset-root data/luxembourgish_corpus/train \\
        --metadata metadata.csv \\
        --output-dir tokens \\
        --tokenizer-path checkpoints/grapheme_mtl_merged_expanded_v1.json \\
        --language-id lb
"""
from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import librosa
import torch
from tqdm import tqdm

from ..models.s3tokenizer import S3Tokenizer, S3_SR
from ..models.tokenizers import MTLTokenizer
from ..models.voice_encoder import VoiceEncoder


logger = logging.getLogger(__name__)


@dataclass
class DataPreparationConfig:
    dataset_root: Path
    metadata: Path = Path("metadata.csv")
    output_dir: Path = Path("tokens")
    path_field: str = "path"
    text_field: str = "text"
    device: str = "cpu"
    tokenizer_path: Optional[Path] = None
    language_id: Optional[str] = None
    max_token_len: Optional[int] = None
    skip_existing: bool = False
    voice_encoder_checkpoint: Optional[Path] = None


@dataclass
class DataPreparationResult:
    total_rows: int
    processed: int
    skipped_audio: int
    skipped_text: int
    output_dir: Path
    max_speech_token_len: int
    max_text_token_len: Optional[int]

    def to_dict(self) -> dict:
        return {
            "total_rows": self.total_rows,
            "processed": self.processed,
            "skipped_audio": self.skipped_audio,
            "skipped_text": self.skipped_text,
            "output_dir": str(self.output_dir),
            "max_speech_token_len": self.max_speech_token_len,
            "max_text_token_len": self.max_text_token_len,
        }


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
        help="Column name for transcriptions. Required when generating text tokens.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for the tokenizer (e.g., cpu, cuda).",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=None,
        help="Path to multilingual tokenizer JSON to emit text tokens. "
             "If omitted, only speech tokens are generated.",
    )
    parser.add_argument(
        "--language-id",
        type=str,
        default=None,
        help="Language code to prepend when generating text tokens (e.g., lb, en, fr).",
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
    parser.add_argument(
        "--voice-encoder-checkpoint",
        type=Path,
        default=None,
        help="Optional path to a voice encoder checkpoint (ve.pt). "
             "When provided, per-sample speaker embeddings are stored in the token files.",
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


def tokenize_text(
    tokenizer: MTLTokenizer,
    text: str,
    language_id: Optional[str],
) -> tuple[torch.Tensor, int]:
    tokens = tokenizer.text_to_tokens(text, language_id=language_id)
    tokens = tokens.squeeze(0)
    return tokens.cpu(), int(tokens.numel())


def run_preparation(config: DataPreparationConfig) -> DataPreparationResult:
    dataset_root = config.dataset_root
    metadata_path = config.metadata if config.metadata.is_absolute() else dataset_root / config.metadata
    output_dir = config.output_dir if config.output_dir.is_absolute() else dataset_root / config.output_dir

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(config.device)

    tokenizer = S3Tokenizer()
    tokenizer.to(device)
    tokenizer.eval()

    text_tokenizer: Optional[MTLTokenizer] = None
    language_id = config.language_id.lower() if config.language_id else None
    if config.tokenizer_path is not None:
        tokenizer_path = config.tokenizer_path if config.tokenizer_path.is_absolute() else dataset_root / config.tokenizer_path
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer JSON not found: {tokenizer_path}")
        text_tokenizer = MTLTokenizer(str(tokenizer_path))
        logger.info("Loaded text tokenizer from %s", tokenizer_path)
        if not config.text_field:
            raise ValueError("text_field must be provided when generating text tokens.")

    voice_encoder: Optional[VoiceEncoder] = None
    ve_path: Optional[Path] = None
    if config.voice_encoder_checkpoint is not None:
        ve_path = (
            config.voice_encoder_checkpoint
            if config.voice_encoder_checkpoint.is_absolute()
            else dataset_root / config.voice_encoder_checkpoint
        )
    else:
        candidate = Path(__file__).resolve().parents[3] / "models" / "multilingual" / "ve.pt"
        if candidate.exists():
            ve_path = candidate
    if ve_path is not None:
        if not ve_path.exists():
            raise FileNotFoundError(f"Voice encoder checkpoint not found: {ve_path}")
        voice_encoder = VoiceEncoder()
        state = torch.load(ve_path, map_location="cpu", weights_only=True)
        voice_encoder.load_state_dict(state)
        voice_encoder.to(device)
        voice_encoder.eval()
        logger.info("Loaded voice encoder from %s", ve_path)
    else:
        logger.warning("Voice encoder checkpoint not provided; speaker embeddings will be zero-padded.")

    logger.info("Loaded S3Tokenizer on %s", device)
    logger.info("Writing token files to %s", output_dir)

    total_rows = 0
    processed = 0
    skipped_audio = 0
    skipped_text = 0
    max_speech_len = 0
    max_text_len: Optional[int] = None

    with metadata_path.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        if config.path_field not in reader.fieldnames:
            raise KeyError(f"Column '{config.path_field}' not found in {metadata_path}")

        for row in tqdm(reader, desc=f"Tokenizing {dataset_root.name}", dynamic_ncols=True):
            total_rows += 1
            audio_rel = Path(row[config.path_field])
            audio_path = dataset_root / audio_rel
            if not audio_path.exists():
                logger.warning("Missing audio file: %s", audio_path)
                skipped_audio += 1
                continue

            sample_id = audio_rel.stem
            out_path = output_dir / f"{sample_id}.pt"
            if config.skip_existing and out_path.exists():
                processed += 1
                try:
                    saved = torch.load(out_path, map_location="cpu")
                    max_speech_len = max(max_speech_len, int(saved.get("speech_token_len", 0)))
                    if "text_token_len" in saved:
                        saved_len = int(saved["text_token_len"])
                        max_text_len = saved_len if max_text_len is None else max(max_text_len, saved_len)
                except Exception as exc:  # pylint: disable=broad-except
                    logger.warning("Failed to read existing token file %s: %s", out_path, exc)
                continue

            audio = load_audio(audio_path)
            audio_np = audio.numpy()
            audio = audio.to(device)
            speech_tokens, speech_len = tokenize_audio(tokenizer, audio, max_len=config.max_token_len)
            max_speech_len = max(max_speech_len, speech_len)

            payload = {
                "speech_tokens": speech_tokens,
                "speech_token_len": speech_len,
                "audio_path": str(audio_rel),
            }

            text_value: Optional[str] = None
            if config.text_field and config.text_field in row:
                text_value = row[config.text_field]
                payload["text"] = text_value

            if text_tokenizer is not None:
                if not text_value:
                    skipped_text += 1
                    logger.warning("Missing text entry for %s; skipping text tokens.", audio_rel)
                else:
                    text_tokens, text_len = tokenize_text(text_tokenizer, text_value, language_id)
                    payload["text_tokens"] = text_tokens
                    payload["text_token_len"] = text_len
                    if language_id:
                        payload["language_id"] = language_id
                    max_text_len = text_len if max_text_len is None else max(max_text_len, text_len)

            if voice_encoder is not None:
                try:
                    embedding = voice_encoder.embeds_from_wavs(
                        [audio_np],
                        sample_rate=S3_SR,
                        as_spk=True,
                    )
                    payload["speaker_embedding"] = torch.from_numpy(embedding).float()
                except Exception as exc:  # pylint: disable=broad-except
                    logger.warning("Failed to compute speaker embedding for %s: %s", audio_rel, exc)

            torch.save(payload, out_path)
            processed += 1

    return DataPreparationResult(
        total_rows=total_rows,
        processed=processed,
        skipped_audio=skipped_audio,
        skipped_text=skipped_text,
        output_dir=output_dir,
        max_speech_token_len=max_speech_len,
        max_text_token_len=max_text_len,
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    config = DataPreparationConfig(
        dataset_root=args.dataset_root,
        metadata=args.metadata,
        output_dir=args.output_dir,
        path_field=args.path_field,
        text_field=args.text_field,
        device=args.device,
        tokenizer_path=args.tokenizer_path,
        language_id=args.language_id,
        max_token_len=args.max_token_len,
        skip_existing=args.skip_existing,
        voice_encoder_checkpoint=args.voice_encoder_checkpoint,
    )
    result = run_preparation(config)

    logger.info(
        "Processed %s/%s rows (skipped audio: %s, skipped text: %s). Max speech tokens: %s. Max text tokens: %s.",
        result.processed,
        result.total_rows,
        result.skipped_audio,
        result.skipped_text,
        result.max_speech_token_len,
        result.max_text_token_len if result.max_text_token_len is not None else "N/A",
    )


if __name__ == "__main__":
    main()
