#!/usr/bin/env python3

"""
Download the Luxembourgish LOD male TTS dataset and build a train/test split.

Creates the folder structure:

luxembourgish_male_corpus/
└── {train,test}/
    ├── audio/
    └── metadata.csv

Metadata schema:
- path: relative path (audio/<filename>)
- text: transcription string
- source: origin identifier for traceability
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from huggingface_hub import snapshot_download


MALE_TTS_REPO = "denZLS/Luxembourgish-Male-TTS-for-LOD"


@dataclass
class Sample:
    source: str
    audio_path: Path
    text: str


def ensure_unzipped(zip_path: Path, target_dir: Path) -> Path:
    """Unpack a zip archive if the target folder is absent."""
    if target_dir.exists():
        return target_dir

    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(target_dir)
    return target_dir


def prepare_male_tts_dataset(work_dir: Path, token: str | None) -> Path:
    """Download and unpack the male TTS dataset."""
    base_dir = work_dir / "luxembourgish_male_tts"
    if (base_dir / "wavs").exists():
        return base_dir
    base_dir.mkdir(parents=True, exist_ok=True)

    archive_path = base_dir / "LOD-male-dataset.zip"

    if not archive_path.exists():
        snapshot_download(
            repo_id=MALE_TTS_REPO,
            repo_type="dataset",
            local_dir=base_dir,
            token=token,
        )
    elif not (base_dir / "extracted").exists():
        print(f"[INFO] Using existing archive at {archive_path}")

    ensure_unzipped(archive_path, base_dir / "extracted")
    wavs_dir = base_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)
    extracted_wavs = base_dir / "extracted" / "LOD-male-dataset" / "wavs"
    if extracted_wavs.exists():
        for src in extracted_wavs.iterdir():
            target = wavs_dir / src.name
            if not target.exists():
                shutil.move(str(src), target)
        shutil.rmtree(extracted_wavs.parent.parent)

    return base_dir


def load_tts_samples(csv_path: Path, wavs_dir: Path, source_prefix: str) -> list[Sample]:
    """Load samples from TTS CSV (pipe-delimited)."""
    rows: list[tuple[str, str]] = []
    for encoding in ("utf-8", "latin-1"):
        try:
            with csv_path.open("r", encoding=encoding) as handle:
                reader = csv.reader(handle, delimiter="|")
                for row in reader:
                    if len(row) < 2:
                        continue
                    file_id, text = row[0].strip(), row[1].strip()
                    if file_id and text:
                        rows.append((file_id, text))
            break
        except UnicodeDecodeError:
            if encoding == "latin-1":
                raise
            continue

    available_files = {p.stem: p for p in wavs_dir.glob("*.wav")}

    samples: list[Sample] = []
    missing_audio: list[str] = []
    for file_id, text in rows:
        audio_path = available_files.get(file_id)
        if audio_path is not None:
            samples.append(Sample(source=source_prefix, audio_path=audio_path, text=text))
        else:
            missing_audio.append(file_id)

    if missing_audio:
        # Allow a fallback when identifiers never match filenames but counts do.
        if len(samples) == 0 and len(rows) == len(available_files):
            samples = []
            for (_, text), wav_path in zip(rows, sorted(available_files.values())):
                samples.append(Sample(source=source_prefix, audio_path=wav_path, text=text))
        else:
            preview = ", ".join(missing_audio[:5])
            raise RuntimeError(
                f"Missing audio files for {len(missing_audio)} entries (examples: {preview}). "
                "Please ensure text/audio pairs are aligned."
            )

    if not samples:
        return samples

    if len(samples) != len(rows):
        raise RuntimeError(
            f"Aligned {len(samples)} samples but input CSV contains {len(rows)} rows. "
            "Please ensure text/audio pairs are aligned."
        )

    return samples


def split_samples(samples: Sequence[Sample], test_ratio: float, seed: int, source_prefix: str) -> tuple[list[Sample], list[Sample]]:
    """Randomly split samples into train/test with deterministic shuffling."""
    if not samples:
        return [], []

    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    test_count = max(1, int(len(shuffled) * test_ratio))
    test = shuffled[:test_count]
    train = shuffled[test_count:]

    # Update source labels to retain provenance plus split info.
    for sample in train:
        sample.source = f"{source_prefix}_train"
    for sample in test:
        sample.source = f"{source_prefix}_test"

    return train, test


def aggregate_samples(train_samples: Sequence[Sample], test_samples: Sequence[Sample], output_dir: Path, link_audio: bool) -> None:
    """Write audio files and metadata CSVs to the dataset."""
    for split, samples in (("train", train_samples), ("test", test_samples)):
        split_dir = output_dir / split
        if split_dir.exists():
            shutil.rmtree(split_dir)
        audio_dir = split_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        metadata_rows: list[tuple[str, str, str]] = []

        for sample in samples:
            target_name = f"{sample.source}_{sample.audio_path.stem}{sample.audio_path.suffix}"
            target_path = audio_dir / target_name
            suffix = 1
            while target_path.exists():
                target_path = audio_dir / f"{sample.source}_{sample.audio_path.stem}_{suffix}{sample.audio_path.suffix}"
                suffix += 1

            if link_audio:
                if target_path.exists():
                    target_path.unlink()
                os.symlink(sample.audio_path.resolve(), target_path)
            else:
                shutil.copy2(sample.audio_path, target_path)

            metadata_rows.append((f"audio/{target_path.name}", sample.text, sample.source))

        csv_path = split_dir / "metadata.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["path", "text", "source"])
            writer.writerows(metadata_rows)


def prepare_dataset(work_dir: Path, token: str | None, link_audio: bool, seed: int, test_ratio: float) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    downloads_dir = work_dir / "artifacts"

    dataset_dir = prepare_male_tts_dataset(downloads_dir, token)
    samples = load_tts_samples(dataset_dir / "LOD-male.csv", dataset_dir / "wavs", "tts_male")
    if not samples:
        raise RuntimeError("No samples could be loaded from the LOD male dataset.")

    train_samples, test_samples = split_samples(samples, test_ratio, seed, "tts_male")
    if not train_samples:
        raise RuntimeError("Not enough samples to build a training set.")

    aggregate_samples(train_samples, test_samples, work_dir, link_audio=link_audio)

    print(f"Male-only dataset written to {work_dir}")
    print(f"Train samples: {len(train_samples)} | Test samples: {len(test_samples)}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the LOD male TTS dataset and build a train/test split.")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("data") / "luxembourgish_male_corpus",
        help="Workspace directory for downloads and outputs.",
    )
    parser.add_argument(
        "--copy-audio",
        action="store_true",
        help="Copy audio files instead of creating symlinks in the dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for dataset splitting.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Proportion of samples assigned to the test split.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    token = os.environ.get("HF_TOKEN")

    try:
        prepare_dataset(
            work_dir=args.work_dir,
            token=token,
            link_audio=not args.copy_audio,
            seed=args.seed,
            test_ratio=args.test_ratio,
        )
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
