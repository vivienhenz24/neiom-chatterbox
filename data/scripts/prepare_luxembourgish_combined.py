#!/usr/bin/env python3

"""
Download Luxembourgish speech datasets and aggregate them into a single split.

Creates the folder structure:

luxembourgish_corpus/
└── {train,test}/
    ├── audio/
    └── metadata.csv

Metadata schema:
- path: relative path (audio/<filename>)
- text: transcription string
- source: origin identifier for traceability

Requirements:
- requests
- huggingface_hub
- numpy

The Hugging Face token is read from the HF_TOKEN environment variable if set.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
import shutil
import sys
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence

import requests
from huggingface_hub import hf_hub_download, snapshot_download, hf_hub_url
from datasets import load_dataset
import numpy as np
import wave


HF_RTL_FILE_ID = "1IiFV6TZHH1sOBL409VnmxCXSSyQkue0F"
FLEURS_REPO = "google/fleurs"
FLEURS_PREFIX = "data/lb_lu"
FLEURS_SPLITS = ("train", "dev", "test")
FEMALE_TTS_REPO = "denZLS/Luxembourgish-Female-TTS-for-LOD"
MALE_TTS_REPO = "denZLS/Luxembourgish-Male-TTS-for-LOD"


@dataclass
class Sample:
    source: str
    audio_path: Path
    text: str


def download_google_drive(file_id: str, destination: Path, chunk_size: int = 1 << 20) -> None:
    """Download a Google Drive file handling the confirmation token."""
    if destination.exists():
        return

    session = requests.Session()
    url = "https://docs.google.com/uc?export=download"
    response = session.get(url, params={"id": file_id}, stream=True)
    response.raise_for_status()

    token = _extract_confirm_token(response)
    if token:
        response.close()
        response = session.get(url, params={"id": file_id, "confirm": token}, stream=True)
        response.raise_for_status()
    else:
        content_type = response.headers.get("Content-Type", "")
        if content_type.startswith("text/html"):
            raise RuntimeError("Failed to obtain confirmation token from Google Drive.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                handle.write(chunk)


def _extract_confirm_token(response: requests.Response) -> str | None:
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    content_type = response.headers.get("Content-Type", "")
    if content_type.startswith("text/html"):
        text = response.text
        match = re.search(r"confirm=([0-9A-Za-z_]+)", text)
        if match:
            return match.group(1)

    return None


def ensure_unzipped(zip_path: Path, target_dir: Path) -> Path:
    """Unpack a zip archive if the target folder is absent."""
    if target_dir.exists():
        return target_dir

    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(target_dir)
    return target_dir


def ensure_untarred(tar_path: Path, target_dir: Path) -> Path:
    """Unpack a tar.gz archive if the target folder is absent."""
    if target_dir.exists():
        return target_dir

    target_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as archive:
        archive.extractall(target_dir)
    return target_dir


def _flatten_wav_directory(root: Path) -> None:
    """
    Move nested WAV files directly under `root`.

    Some archives include redundant directory levels (e.g. train/train/*.wav).
    """
    if any(root.glob("*.wav")):
        return

    wavs = sorted(root.rglob("*.wav"))
    for wav in wavs:
        target = root / wav.name
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(wav), target)

    # Remove empty directories left behind
    for subdir in sorted(root.rglob("*"), reverse=True):
        if subdir.is_dir():
            try:
                subdir.rmdir()
            except OSError:
                continue


def _ensure_valid_tar(path: Path, expected_size: int | None = None) -> bool:
    """Return True if path exists, matches expected size, and is a readable tar archive."""
    if not path.exists():
        return False
    if expected_size and path.stat().st_size != expected_size:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return False
    if tarfile.is_tarfile(path):
        try:
            with tarfile.open(path, "r:gz") as archive:
                for _ in archive:
                    pass
            return True
        except tarfile.TarError:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            return False
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    return False


def _get_hf_file_size(repo_id: str, filename: str, token: str | None, repo_type: str = "dataset") -> int:
    url = hf_hub_url(repo_id=repo_id, filename=filename, repo_type=repo_type)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with requests.Session() as session:
        head = session.head(url, headers=headers, allow_redirects=True)
        head.raise_for_status()
        return int(head.headers.get("Content-Length", 0))


def _download_hf_file(repo_id: str, filename: str, destination: Path, token: str | None, repo_type: str = "dataset") -> None:
    """Download a large file from Hugging Face with manual range requests."""
    url = hf_hub_url(repo_id=repo_id, filename=filename, repo_type=repo_type)
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".download")

    with requests.Session() as session:
        head = session.head(url, headers=headers, allow_redirects=True)
        head.raise_for_status()
        total = int(head.headers.get("Content-Length", 0))

        if total <= 0:
            with session.get(url, headers=headers, stream=True, allow_redirects=True) as resp, tmp_path.open("wb") as handle:
                resp.raise_for_status()
                for chunk in resp.iter_content(1 << 20):
                    if chunk:
                        handle.write(chunk)
        else:
            chunk_size = 1 << 26  # 64 MiB
            downloaded = 0
            with tmp_path.open("wb") as handle:
                while downloaded < total:
                    end = min(downloaded + chunk_size - 1, total - 1)
                    range_headers = dict(headers)
                    range_headers["Range"] = f"bytes={downloaded}-{end}"
                    with session.get(url, headers=range_headers, stream=True, allow_redirects=True) as resp:
                        resp.raise_for_status()
                        for chunk in resp.iter_content(1 << 20):
                            if chunk:
                                handle.write(chunk)
                                downloaded += len(chunk)

    tmp_path.replace(destination)

    suffixes = destination.suffixes[-2:]
    if suffixes == [".tar", ".gz"]:
        try:
            with tarfile.open(destination, "r:gz") as archive:
                for _ in archive:
                    pass
        except tarfile.TarError as exc:  # pragma: no cover
            try:
                destination.unlink()
            finally:
                raise RuntimeError(f"Corrupted download detected for {filename}") from exc


def _count_tsv_records(tsv_path: Path) -> int:
    with tsv_path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle) - 1  # subtract header


def _populate_fleurs_from_dataset(target_dir: Path, split: str, token: str | None) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(
        "google/fleurs",
        "lb_lu",
        split=split,
        token=token,
        download_mode="reuse_dataset_if_exists",
        trust_remote_code=True,
    )
    for sample in ds:
        audio_info = sample["audio"]
        filename = Path(audio_info.get("path", "")).name or f"{sample['id']}.wav"
        destination = target_dir / filename
        if destination.exists():
            continue
        array = audio_info["array"]
        if array is None:
            continue
        samples = np.clip(array, -1.0, 1.0)
        int_samples = (samples * 32767.0).astype(np.int16)
        channels = 1
        data = int_samples
        if int_samples.ndim == 2:
            channels = int_samples.shape[1]
            data = int_samples.reshape(-1)
        with wave.open(str(destination), "wb") as handle:
            handle.setnchannels(channels)
            handle.setsampwidth(2)
            handle.setframerate(audio_info["sampling_rate"])
            handle.writeframes(data.tobytes())


def prepare_rtl_dataset(work_dir: Path, fallback_archive: Path | None = None) -> Path:
    """
    Download and extract the RTL benchmark dataset.

    Returns the path containing dev/ and test/ folders.
    """
    target_dir = work_dir / "rtl_dataset"
    if (target_dir / "dev").exists() and (target_dir / "test").exists():
        return target_dir

    archive_path = work_dir / "rtl_benchmark.zip"
    if archive_path.exists():
        pass
    elif fallback_archive and fallback_archive.exists():
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fallback_archive, archive_path)
    else:
        download_google_drive(HF_RTL_FILE_ID, archive_path)

    temp_extract = work_dir / "rtl_benchmark_raw"
    ensure_unzipped(archive_path, temp_extract)

    target_dir.mkdir(parents=True, exist_ok=True)
    for split in ("dev", "test", "train"):
        src = temp_extract / split
        if src.exists():
            shutil.copytree(src, target_dir / split, dirs_exist_ok=True)

    return target_dir


def prepare_fleurs_dataset(work_dir: Path, token: str | None) -> Path:
    """
    Download the Luxembourgish (lb_lu) portion of the FLEURS dataset.

    Returns path containing split subfolders with extracted WAV files.
    """
    base_dir = work_dir / "fleurs_lb_lu"
    audio_dir = base_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    for split in FLEURS_SPLITS:
        tsv_path = base_dir / f"{split}.tsv"
        if not tsv_path.exists():
            downloaded = Path(
                hf_hub_download(
                    repo_id=FLEURS_REPO,
                    filename=f"{FLEURS_PREFIX}/{split}.tsv",
                    repo_type="dataset",
                    local_dir=base_dir,
                    local_dir_use_symlinks=False,
                    token=token,
                )
            )

            tsv_path.parent.mkdir(parents=True, exist_ok=True)
            if downloaded.resolve() != tsv_path.resolve():
                shutil.move(str(downloaded), tsv_path)

        archive_name = f"{split}.tar.gz"
        archive_path = audio_dir / archive_name
        expected_size = _get_hf_file_size(
            repo_id=FLEURS_REPO,
            filename=f"{FLEURS_PREFIX}/audio/{archive_name}",
            token=token,
        )
        need_download = not _ensure_valid_tar(archive_path, expected_size=expected_size)
        if need_download:
            attempts = 0
            while True:
                attempts += 1
                try:
                    _download_hf_file(
                        repo_id=FLEURS_REPO,
                        filename=f"{FLEURS_PREFIX}/audio/{archive_name}",
                        destination=archive_path,
                        token=token,
                    )
                    break
                except RuntimeError:
                    if attempts >= 3:
                        raise
                    print(f"[WARN] Retry {attempts} downloading {archive_name}")

        extracted = audio_dir / split
        if need_download and extracted.exists():
            shutil.rmtree(extracted)
        ensure_untarred(archive_path, extracted)
        _flatten_wav_directory(extracted)

        expected_count = max(_count_tsv_records(tsv_path), 0)
        current_count = len(list(extracted.glob("*.wav")))
        if expected_count and current_count < expected_count:
            dataset_split = "validation" if split == "dev" else split
            _populate_fleurs_from_dataset(extracted, dataset_split, token)

    return base_dir

def prepare_female_tts_dataset(work_dir: Path, token: str | None) -> Path:
    """Download and unpack the female TTS dataset."""
    base_dir = work_dir / "luxembourgish_female_tts"
    if (base_dir / "wavs").exists():
        return base_dir

    snapshot_download(
        repo_id=FEMALE_TTS_REPO,
        repo_type="dataset",
        local_dir=base_dir,
        token=token,
    )

    archive_path = base_dir / "LOD-female-dataset.zip"
    ensure_unzipped(archive_path, base_dir / "extracted")
    wavs_dir = base_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)
    extracted_wavs = base_dir / "extracted" / "wavs"
    if extracted_wavs.exists():
        for src in extracted_wavs.iterdir():
            target = wavs_dir / src.name
            if not target.exists():
                shutil.move(str(src), target)
        shutil.rmtree(extracted_wavs.parent)

    return base_dir


def prepare_male_tts_dataset(work_dir: Path, token: str | None) -> Path:
    """Download and unpack the male TTS dataset."""
    base_dir = work_dir / "luxembourgish_male_tts"
    if (base_dir / "wavs").exists():
        return base_dir

    snapshot_download(
        repo_id=MALE_TTS_REPO,
        repo_type="dataset",
        local_dir=base_dir,
        token=token,
    )

    archive_path = base_dir / "LOD-male-dataset.zip"
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


def load_rtl_samples(rtl_dir: Path) -> tuple[list[Sample], list[Sample]]:
    """Load dev (train) and test samples from the RTL dataset."""
    train_samples = _read_tsv_samples(
        rtl_dir / "dev" / "dev.tsv",
        rtl_dir / "dev",
        source_prefix="rtl_dev",
    )

    test_samples = _read_tsv_samples(
        rtl_dir / "test" / "test.tsv",
        rtl_dir / "test",
        source_prefix="rtl_test",
    )

    return train_samples, test_samples


def _read_tsv_samples(tsv_path: Path, audio_dir: Path, source_prefix: str) -> list[Sample]:
    samples: list[Sample] = []
    if not tsv_path.exists():
        return samples

    with tsv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or "filename" not in reader.fieldnames or "transcription" not in reader.fieldnames:
            raise ValueError(f"Unexpected TSV format: {tsv_path}")

        for row in reader:
            filename = row["filename"].strip()
            text = row["transcription"].strip()
            if not filename or not text:
                continue
            audio_path = audio_dir / filename
            if audio_path.exists():
                samples.append(Sample(source=source_prefix, audio_path=audio_path, text=text))
    return samples


def load_fleurs_samples(fleurs_dir: Path) -> tuple[list[Sample], list[Sample]]:
    """Load FLEURS train/dev/test as train/test splits."""
    train_samples = _read_fleurs_split(fleurs_dir, "train", source_prefix="fleurs_train")
    dev_samples = _read_fleurs_split(fleurs_dir, "dev", source_prefix="fleurs_dev")
    test_samples = _read_fleurs_split(fleurs_dir, "test", source_prefix="fleurs_test")

    train = train_samples
    test = dev_samples + test_samples
    return train, test


def _read_fleurs_split(base_dir: Path, split: str, source_prefix: str) -> list[Sample]:
    tsv_path = base_dir / f"{split}.tsv"
    audio_dir = base_dir / "audio" / split
    samples: list[Sample] = []

    with tsv_path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row:
                continue
            if row[0].lower() == "id":
                continue
            if len(row) < 3:
                continue
            filename = row[1].strip()
            text = row[2].strip()
            if not filename or not text:
                continue
            audio_path = audio_dir / Path(filename).name
            if audio_path.exists():
                samples.append(Sample(source=source_prefix, audio_path=audio_path, text=text))
    return samples


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
    for file_id, text in rows:
        audio_path = available_files.get(file_id)
        if audio_path is not None:
            samples.append(Sample(source=source_prefix, audio_path=audio_path, text=text))

    if samples:
        return samples

    # Fallback: align by sorted order if identifiers do not match filenames.
    wav_list = sorted(available_files.values())
    if len(wav_list) != len(rows):
        return samples  # Unable to align safely

    for (file_id, text), wav_path in zip(rows, wav_list):
        samples.append(Sample(source=source_prefix, audio_path=wav_path, text=text))

    return samples


def split_samples(samples: Sequence[Sample], test_ratio: float, seed: int, source_prefix: str) -> tuple[list[Sample], list[Sample]]:
    """Randomly split samples into train/test with deterministic shuffling."""
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
    """Write audio files and metadata CSVs to the combined dataset."""
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


def prepare_datasets(work_dir: Path, token: str | None, link_audio: bool, seed: int, female_test_ratio: float, male_test_ratio: float) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)

    fallback_rtl = Path("data") / "benchmark.zip"
    rtl_dir = prepare_rtl_dataset(work_dir / "rtl", fallback_archive=fallback_rtl)
    fleurs_dir = prepare_fleurs_dataset(work_dir / "fleurs", token)
    female_dir = prepare_female_tts_dataset(work_dir / "female_tts", token)
    male_dir = prepare_male_tts_dataset(work_dir / "male_tts", token)

    rtl_train, rtl_test = load_rtl_samples(rtl_dir)
    fleurs_train, fleurs_test = load_fleurs_samples(fleurs_dir)
    female_samples = load_tts_samples(female_dir / "LOD-female.csv", female_dir / "wavs", "tts_female")
    male_samples = load_tts_samples(male_dir / "LOD-male.csv", male_dir / "wavs", "tts_male")

    female_train, female_test = split_samples(female_samples, female_test_ratio, seed, "tts_female")
    male_train, male_test = split_samples(male_samples, male_test_ratio, seed, "tts_male")

    combined_train = rtl_train + fleurs_train + female_train + male_train
    combined_test = rtl_test + fleurs_test + female_test + male_test

    output_dir = work_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_samples(combined_train, combined_test, output_dir, link_audio=link_audio)

    print(f"Combined dataset written to {output_dir}")
    print(f"Train samples: {len(combined_train)} | Test samples: {len(combined_test)}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and combine Luxembourgish speech datasets.")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("data") / "luxembourgish_corpus",
        help="Workspace directory for downloads and outputs.",
    )
    parser.add_argument(
        "--copy-audio",
        action="store_true",
        help="Copy audio files instead of creating symlinks in the combined dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for dataset splitting.",
    )
    parser.add_argument(
        "--female-test-ratio",
        type=float,
        default=0.1,
        help="Proportion of female TTS samples assigned to the test split.",
    )
    parser.add_argument(
        "--male-test-ratio",
        type=float,
        default=0.1,
        help="Proportion of male TTS samples assigned to the test split.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    token = os.environ.get("HF_TOKEN")

    try:
        prepare_datasets(
            work_dir=args.work_dir,
            token=token,
            link_audio=not args.copy_audio,
            seed=args.seed,
            female_test_ratio=args.female_test_ratio,
            male_test_ratio=args.male_test_ratio,
        )
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
def _download_hf_file(repo_id: str, filename: str, destination: Path, token: str | None, repo_type: str = "dataset") -> None:
    """Download a large file from Hugging Face with manual range requests."""
    url = hf_hub_url(repo_id=repo_id, filename=filename, repo_type=repo_type)
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".download")

    with requests.Session() as session:
        head = session.head(url, headers=headers, allow_redirects=True)
        head.raise_for_status()
        total = int(head.headers.get("Content-Length", 0))

        if total <= 0:
            # Fallback to simple streaming GET without range headers
            with session.get(url, headers=headers, stream=True, allow_redirects=True) as resp, tmp_path.open("wb") as handle:
                resp.raise_for_status()
                for chunk in resp.iter_content(1 << 20):
                    if chunk:
                        handle.write(chunk)
        else:
            chunk_size = 1 << 26  # 64 MiB
            downloaded = 0
            with tmp_path.open("wb") as handle:
                while downloaded < total:
                    end = min(downloaded + chunk_size - 1, total - 1)
                    range_headers = dict(headers)
                    range_headers["Range"] = f"bytes={downloaded}-{end}"
                    with session.get(url, headers=range_headers, stream=True, allow_redirects=True) as resp:
                        resp.raise_for_status()
                        for chunk in resp.iter_content(1 << 20):
                            if chunk:
                                handle.write(chunk)
                                downloaded += len(chunk)

        tmp_path.replace(destination)

    # Basic integrity check for tarballs
    suffixes = destination.suffixes[-2:]
    if suffixes == [".tar", ".gz"]:
        try:
            with tarfile.open(destination, "r:gz") as archive:
                for _ in archive:
                    pass
        except tarfile.TarError as exc:  # pragma: no cover
            try:
                destination.unlink()
            finally:
                raise RuntimeError(f"Corrupted download detected for {filename}") from exc


def _count_tsv_records(tsv_path: Path) -> int:
    with tsv_path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle) - 1  # subtract header


def _populate_fleurs_from_dataset(target_dir: Path, split: str, token: str | None) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(
        "google/fleurs",
        "lb_lu",
        split=split,
        token=token,
        download_mode="reuse_dataset_if_exists",
        trust_remote_code=True,
    )
    for sample in ds:
        source_path = Path(sample["path"])
        if not source_path.exists():
            continue
        destination = target_dir / source_path.name
        if not destination.exists():
            shutil.copy2(source_path, destination)
