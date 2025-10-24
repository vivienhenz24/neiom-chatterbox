#!/usr/bin/env python3

from __future__ import annotations

import csv
import sys
from pathlib import Path


def collect_split_directories(base_path: Path) -> list[Path]:
    """Return dataset split directories inside rtl_dataset, sorted by name."""
    return sorted(
        [
            path
            for path in base_path.iterdir()
            if path.is_dir()
        ],
        key=lambda p: p.name,
    )


def find_split_tsv(split_dir: Path) -> Path | None:
    """
    Locate the TSV file describing the split.

    By convention it should match `<split>.tsv`, but we fall back to picking the
    only TSV if the convention is not met.
    """
    conventional = split_dir / f"{split_dir.name}.tsv"
    if conventional.exists():
        return conventional

    candidates = list(split_dir.glob("*.tsv"))
    if len(candidates) == 1:
        return candidates[0]

    return None


def read_tsv_filenames(tsv_path: Path) -> set[str]:
    """Extract the set of filenames from the TSV file."""
    try:
        with tsv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if not reader.fieldnames or "filename" not in reader.fieldnames:
                raise ValueError(
                    f"'filename' column not found in {tsv_path}"
                )
            filenames = {
                row["filename"].strip()
                for row in reader
                if row.get("filename")
            }
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing TSV file: {tsv_path}") from exc

    return filenames


def read_split_files(split_dir: Path) -> set[str]:
    """Return the set of data files (excluding TSVs) present on disk."""
    return {
        path.name
        for path in split_dir.iterdir()
        if path.is_file() and path.suffix.lower() != ".tsv"
    }


def main() -> int:
    base_path = Path(__file__).resolve().parents[1] / "rtl_dataset"
    if not base_path.exists():
        print(f"[ERROR] Dataset directory not found: {base_path}", file=sys.stderr)
        return 1

    all_splits_ok = True

    for split_dir in collect_split_directories(base_path):
        tsv_path = find_split_tsv(split_dir)
        if tsv_path is None:
            print(f"[ERROR] No TSV file found for split '{split_dir.name}'")
            all_splits_ok = False
            continue

        try:
            tsv_filenames = read_tsv_filenames(tsv_path)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[ERROR] {exc}")
            all_splits_ok = False
            continue

        disk_filenames = read_split_files(split_dir)

        missing_in_tsv = sorted(disk_filenames - tsv_filenames)
        missing_on_disk = sorted(tsv_filenames - disk_filenames)

        if not missing_in_tsv and not missing_on_disk:
            print(f"[OK] {split_dir.name}: all files match {tsv_path.name}")
            continue

        all_splits_ok = False
        print(f"[MISMATCH] {split_dir.name}:")

        if missing_in_tsv:
            print("  Files missing in TSV:")
            for name in missing_in_tsv:
                print(f"    {name}")

        if missing_on_disk:
            print("  TSV entries missing on disk:")
            for name in missing_on_disk:
                print(f"    {name}")

    return 0 if all_splits_ok else 1


if __name__ == "__main__":
    sys.exit(main())

