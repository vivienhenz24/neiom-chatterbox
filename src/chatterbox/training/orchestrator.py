"""
High-level orchestration for Luxembourgish T3 fine-tuning data preparation.

This module stitches together the raw corpus builder located under
`data/scripts/prepare_luxembourgish_combined.py` and the tokenization step defined in
`chatterbox.training.data_preparation`.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from importlib import util as importlib_util
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

from .data_preparation import (
    DataPreparationConfig,
    DataPreparationResult,
    run_preparation,
)

logger = logging.getLogger(__name__)

_LUX_MODULE = None


def _load_lux_module():
    global _LUX_MODULE  # pylint: disable=global-statement
    if _LUX_MODULE is not None:
        return _LUX_MODULE

    script_path = Path(__file__).resolve().parents[2] / "data" / "scripts" / "prepare_luxembourgish_combined.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Luxembourgish preparation script not found at {script_path}")

    spec = importlib_util.spec_from_file_location("chatterbox.prepare_luxembourgish_combined", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to create import spec for {script_path}")

    module = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    _LUX_MODULE = module
    return module


def _serialize(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(v) for v in value]
    return value


def _serialize_dataclass(instance) -> dict:
    return _serialize(asdict(instance))


@dataclass
class CorpusStageConfig:
    work_dir: Path = Path("data") / "luxembourgish_corpus"
    copy_audio: bool = False
    seed: int = 42
    female_test_ratio: float = 0.1
    male_test_ratio: float = 0.1


@dataclass
class TokenizationStageConfig:
    tokenizer_path: Optional[Path] = None
    device: str = "cpu"
    language_id: Optional[str] = None
    output_subdir: Path = Path("tokens")
    path_field: str = "path"
    text_field: str = "text"
    max_token_len: Optional[int] = None
    skip_existing: bool = False


@dataclass
class PipelineConfig:
    corpus: CorpusStageConfig = field(default_factory=CorpusStageConfig)
    tokenization: TokenizationStageConfig = field(default_factory=TokenizationStageConfig)
    splits: Sequence[str] = ("train", "test")
    manifest_path: Optional[Path] = None


class LuxembourgishPipeline:
    def __init__(self, config: PipelineConfig, hf_token: Optional[str] = None):
        self.config = config
        self.hf_token = hf_token if hf_token is not None else os.getenv("HF_TOKEN")
        self._prepare_module = _load_lux_module()

    def prepare_corpus(self) -> None:
        cfg = self.config.corpus
        prepare_datasets = getattr(self._prepare_module, "prepare_datasets", None)
        if prepare_datasets is None:
            raise AttributeError("prepare_luxembourgish_combined.prepare_datasets not available")

        logger.info("Preparing Luxembourgish corpus in %s", cfg.work_dir)
        prepare_datasets(
            work_dir=cfg.work_dir,
            token=self.hf_token,
            link_audio=not cfg.copy_audio,
            seed=cfg.seed,
            female_test_ratio=cfg.female_test_ratio,
            male_test_ratio=cfg.male_test_ratio,
        )

    def tokenize(self) -> Dict[str, DataPreparationResult]:
        token_cfg = self.config.tokenization
        if token_cfg.tokenizer_path is not None:
            tokenizer_path = token_cfg.tokenizer_path
            if not tokenizer_path.is_absolute():
                tokenizer_path = tokenizer_path.resolve(strict=False)
        else:
            tokenizer_path = None

        results: Dict[str, DataPreparationResult] = {}
        for split in self.config.splits:
            split_root = self.config.corpus.work_dir / split
            metadata_path = split_root / "metadata.csv"
            if not metadata_path.exists():
                raise FileNotFoundError(f"Metadata for split '{split}' not found at {metadata_path}")

            cfg = DataPreparationConfig(
                dataset_root=split_root,
                metadata=Path("metadata.csv"),
                output_dir=token_cfg.output_subdir,
                path_field=token_cfg.path_field,
                text_field=token_cfg.text_field,
                device=token_cfg.device,
                tokenizer_path=tokenizer_path,
                language_id=token_cfg.language_id,
                max_token_len=token_cfg.max_token_len,
                skip_existing=token_cfg.skip_existing,
            )
            logger.info("Tokenizing split '%s'", split)
            results[split] = run_preparation(cfg)
        return results

    def write_manifest(self, results: Dict[str, DataPreparationResult]) -> Path:
        manifest_path = self.config.manifest_path
        if manifest_path is None:
            manifest_path = self.config.corpus.work_dir / "manifest.json"
        elif not manifest_path.is_absolute():
            manifest_path = self.config.corpus.work_dir / manifest_path

        manifest = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "hf_token_present": bool(self.hf_token),
            "corpus": _serialize_dataclass(self.config.corpus),
            "tokenization": _serialize_dataclass(self.config.tokenization),
            "splits": {split: result.to_dict() for split, result in results.items()},
        }

        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2))
        return manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Luxembourgish T3 training pipeline orchestrator.")
    parser.add_argument("--work-dir", type=Path, default=CorpusStageConfig().work_dir, help="Workspace directory.")
    parser.add_argument("--copy-audio", action="store_true", help="Copy audio instead of symlinking when combining datasets.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset splitting.")
    parser.add_argument("--female-test-ratio", type=float, default=0.1, help="Fraction of female TTS samples used for test.")
    parser.add_argument("--male-test-ratio", type=float, default=0.1, help="Fraction of male TTS samples used for test.")

    parser.add_argument("--tokenizer-path", type=Path, default=None, help="Path to multilingual tokenizer JSON.")
    parser.add_argument("--language-id", type=str, default=None, help="Language code for text tokenization (e.g., lb).")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for tokenization.")
    parser.add_argument("--max-token-len", type=int, default=None, help="Optional cap on speech token sequence length.")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse previously generated token files when present.")
    parser.add_argument("--output-subdir", type=Path, default=Path("tokens"), help="Subdirectory to store token files.")
    parser.add_argument("--path-field", type=str, default="path", help="Metadata column containing audio paths.")
    parser.add_argument("--text-field", type=str, default="text", help="Metadata column containing transcripts.")

    parser.add_argument("--splits", nargs="+", default=["train", "test"], help="Dataset splits to tokenize.")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional manifest output path.")
    parser.add_argument("--no-corpus", action="store_true", help="Skip corpus preparation stage.")
    parser.add_argument("--no-tokenization", action="store_true", help="Skip tokenization stage.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, ...).")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="[%(levelname)s] %(message)s")

    corpus_cfg = CorpusStageConfig(
        work_dir=args.work_dir,
        copy_audio=args.copy_audio,
        seed=args.seed,
        female_test_ratio=args.female_test_ratio,
        male_test_ratio=args.male_test_ratio,
    )
    token_cfg = TokenizationStageConfig(
        tokenizer_path=args.tokenizer_path,
        device=args.device,
        language_id=args.language_id,
        output_subdir=args.output_subdir,
        path_field=args.path_field,
        text_field=args.text_field,
        max_token_len=args.max_token_len,
        skip_existing=args.skip_existing,
    )
    pipeline_cfg = PipelineConfig(
        corpus=corpus_cfg,
        tokenization=token_cfg,
        splits=tuple(args.splits),
        manifest_path=args.manifest,
    )

    pipeline = LuxembourgishPipeline(pipeline_cfg)

    if not args.no_corpus:
        pipeline.prepare_corpus()
    else:
        logger.info("Skipping corpus preparation stage.")

    results: Dict[str, DataPreparationResult] = {}
    if not args.no_tokenization:
        results = pipeline.tokenize()
        manifest_path = pipeline.write_manifest(results)
        logger.info("Manifest written to %s", manifest_path)
    else:
        logger.info("Skipping tokenization stage.")


if __name__ == "__main__":
    main()
