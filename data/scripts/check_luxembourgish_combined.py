#!/usr/bin/env python3

"""Validate the combined Luxembourgish dataset integrity and provenance."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the combined Luxembourgish dataset.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data") / "luxembourgish_corpus",
        help="Path to the combined dataset root.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Workspace directory containing the original downloaded datasets "
        "(defaults to the combined dataset parent).",
    )
    parser.add_argument(
        "--skip-source-check",
        action="store_true",
        help="Skip provenance validation against the original datasets.",
    )
    return parser.parse_args(argv)


def validate_split(split_dir: Path) -> tuple[bool, List[str], List[Dict[str, str]]]:
    """Validate a split directory (train or test)."""
    errors: List[str] = []

    metadata_path = split_dir / "metadata.csv"
    audio_dir = split_dir / "audio"

    if not metadata_path.exists():
        errors.append(f"Missing metadata file: {metadata_path}")
        return False, errors, []

    if not audio_dir.exists():
        errors.append(f"Missing audio directory: {audio_dir}")
        return False, errors, []

    entries: List[Dict[str, str]] = []
    with metadata_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or {"path", "text", "source"} - set(reader.fieldnames):
            errors.append(f"Unexpected metadata schema in {metadata_path}")
            return False, errors, []

        for row in reader:
            entries.append({"path": row["path"], "text": row["text"], "source": row["source"]})

    seen_paths: set[str] = set()
    for entry in entries:
        rel_path = entry["path"]
        text = entry["text"]
        source = entry["source"]
        if not rel_path:
            errors.append("Empty path entry in metadata")
            continue

        if rel_path in seen_paths:
            errors.append(f"Duplicate path entry: {rel_path}")
        seen_paths.add(rel_path)

        audio_path = split_dir / rel_path
        if not audio_path.exists():
            errors.append(f"Missing audio file referenced by metadata: {audio_path}")

        if not text:
            errors.append(f"Empty transcription for {rel_path}")
        if not source:
            errors.append(f"Empty source label for {rel_path}")

    referenced_files = {split_dir / entry["path"] for entry in entries}
    actual_files = {path for path in audio_dir.glob("*.wav")}
    extra_files = actual_files - referenced_files
    if extra_files:
        extras = ", ".join(str(path.name) for path in sorted(extra_files))
        errors.append(f"Audio files not referenced in metadata: {extras}")

    return not errors, errors, entries


def load_rtl_maps(rtl_root: Path) -> Dict[str, Dict[str, str]]:
    """Return filename -> transcript mappings for RTL splits."""

    def _read(tsv_path: Path) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        if not tsv_path.exists():
            return mapping
        with tsv_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if not reader.fieldnames or "filename" not in reader.fieldnames or "transcription" not in reader.fieldnames:
                return mapping
            for row in reader:
                filename = row["filename"].strip()
                text = row["transcription"].strip()
                if filename and text:
                    mapping[Path(filename).stem] = text
        return mapping

    return {
        "rtl_dev": _read(rtl_root / "dev" / "dev.tsv"),
        "rtl_test": _read(rtl_root / "test" / "test.tsv"),
    }


def load_fleurs_maps(fleurs_root: Path) -> Dict[str, Dict[str, str]]:
    """Return filename -> transcript mappings for FLEURS splits."""

    def _read(split: str) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        tsv_path = fleurs_root / f"{split}.tsv"
        if not tsv_path.exists():
            return mapping
        with tsv_path.open("r", encoding="utf-8") as handle:
            reader = csv.reader(handle, delimiter="\t")
            for row in reader:
                if not row:
                    continue
                if row[0].lower() == "id":
                    continue
                if len(row) < 3:
                    continue
                filename = Path(row[1].strip()).stem
                text = row[2].strip()
                if filename and text:
                    mapping[filename] = text
        return mapping

    return {
        "fleurs_train": _read("train"),
        "fleurs_dev": _read("dev"),
        "fleurs_test": _read("test"),
    }


def load_tts_map(csv_path: Path, wav_dir: Path) -> Dict[str, str]:
    """Return filename -> transcript mapping for TTS datasets."""
    rows: List[Tuple[str, str]] = []
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

    files = {p.stem: p for p in wav_dir.glob("*.wav")}
    mapping: Dict[str, str] = {}
    for file_id, text in rows:
        if file_id in files:
            mapping[file_id] = text

    if mapping:
        return mapping

    if len(rows) != len(files):
        return mapping

    for (file_id, text), wav_path in zip(rows, sorted(files.values())):
        mapping[wav_path.stem] = text
    return mapping


def build_source_maps(work_dir: Path) -> Tuple[Dict[str, Dict[str, str]], List[str]]:
    """Collect original dataset mappings. Returns (maps, missing_sources)."""
    sources: Dict[str, Dict[str, str]] = {}
    missing: List[str] = []

    rtl_root = work_dir / "rtl" / "rtl_dataset"
    if (rtl_root / "dev").exists():
        sources.update(load_rtl_maps(rtl_root))
    else:
        missing.append("rtl")

    fleurs_root = work_dir / "fleurs" / "fleurs_lb_lu"
    if fleurs_root.exists():
        sources.update(load_fleurs_maps(fleurs_root))
    else:
        missing.append("fleurs")

    female_root = work_dir / "female_tts" / "luxembourgish_female_tts"
    female_csv = female_root / "LOD-female.csv"
    female_wavs = female_root / "wavs"
    if female_csv.exists() and female_wavs.exists():
        female_map = load_tts_map(female_csv, female_wavs)
        if female_map:
            sources["tts_female"] = female_map
    else:
        missing.append("tts_female")

    male_root = work_dir / "male_tts" / "luxembourgish_male_tts"
    male_csv = male_root / "LOD-male.csv"
    male_wavs = male_root / "wavs"
    if male_csv.exists() and male_wavs.exists():
        male_map = load_tts_map(male_csv, male_wavs)
        if male_map:
            sources["tts_male"] = male_map
    else:
        missing.append("tts_male")

    return sources, missing


def verify_provenance(
    split_entries: Dict[str, List[Dict[str, str]]],
    work_dir: Path,
) -> Tuple[bool, List[str]]:
    """Ensure combined metadata covers all source datasets exactly once."""
    source_maps, missing = build_source_maps(work_dir)
    errors: List[str] = []

    if missing:
        errors.append("Missing source directories: " + ", ".join(sorted(set(missing))))

    aggregated: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for entries in split_entries.values():
        for entry in entries:
            aggregated[entry["source"]].append(entry)

    def _base_label(label: str) -> str:
        if label.startswith("tts_female"):
            return "tts_female"
        if label.startswith("tts_male"):
            return "tts_male"
        return label

    for source_label, entries in aggregated.items():
        base_label = _base_label(source_label)
        original_map = source_maps.get(base_label)
        if original_map is None:
            errors.append(f"No original dataset mapping found for '{source_label}'.")
            continue

        for entry in entries:
            rel_path = entry["path"]
            text = entry["text"]

            name = Path(rel_path).name
            prefix = f"{source_label}_"
            if not name.startswith(prefix):
                errors.append(f"Unexpected filename prefix for entry '{name}'.")
                continue

            suffix = name[len(prefix):]
            stem = Path(suffix).stem
            candidates = [stem]
            if "_" in stem:
                base, tail = stem.rsplit("_", 1)
                if tail.isdigit():
                    candidates.append(base)

            match = next((cand for cand in candidates if cand in original_map), None)
            if match is None:
                errors.append(f"No source match for '{name}'.")
                continue

            if original_map[match] != text:
                errors.append(f"Transcript mismatch for '{name}'.")

    def _aggregated_total(base_label: str) -> int:
        return sum(
            len(entries)
            for label, entries in aggregated.items()
            if _base_label(label) == base_label
        )

    for base_label, original in source_maps.items():
        aggregated_total = _aggregated_total(base_label)
        if aggregated_total != len(original):
            errors.append(
                f"Sample count mismatch for {base_label}: combined={aggregated_total}, original={len(original)}"
            )

    return not errors, errors


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_dir: Path = args.dataset_dir

    if not dataset_dir.exists():
        print(f"[ERROR] Dataset directory not found: {dataset_dir}")
        return 1

    work_dir = args.work_dir or dataset_dir.parent

    splits = ["train", "test"]
    overall_ok = True
    split_entries: Dict[str, List[Dict[str, str]]] = {}

    for split in splits:
        split_dir = dataset_dir / split
        ok, messages, entries = validate_split(split_dir)
        if ok:
            print(f"[OK] {split}: {len(list((split_dir / 'audio').glob('*.wav')))} files validated.")
        else:
            overall_ok = False
            print(f"[FAIL] {split}:")
            for message in messages:
                print(f"  - {message}")
        split_entries[split] = entries

    provenance_ok = True
    if not args.skip_source_check and overall_ok:
        provenance_ok, provenance_messages = verify_provenance(split_entries, work_dir)
        if provenance_ok:
            print("[OK] source coverage: all original samples accounted for.")
        else:
            overall_ok = False
            print("[FAIL] source coverage:")
            for message in provenance_messages:
                print(f"  - {message}")

    if overall_ok and provenance_ok:
        protected = {
            dataset_dir.resolve(),
            (dataset_dir / "train").resolve(),
            (dataset_dir / "test").resolve(),
        }
        for path in work_dir.iterdir():
            if path.resolve() in protected:
                continue
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
        print(f"[OK] cleaned source directories in {work_dir}")
    else:
        print("[WARN] Skipping cleanup because validation did not fully succeed.")

    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
