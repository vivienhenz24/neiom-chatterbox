"""
CLI entry point for fine-tuning the T3 model.
"""
from __future__ import annotations

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch

from .config import T3FineTuningConfig, apply_overrides, load_config
from .datasets.t3_dataset import T3TokenDataset
from .eval import PromptSpec, evaluate_losses, generate_qualitative_samples
from .model_utils import (
    build_optimizer,
    build_scheduler,
    create_grad_scaler,
    load_multilingual_t3,
    load_s3gen,
)
from .trainer import Trainer
from .utils import ensure_directory, seed_everything, setup_logging, timed_block

logger = logging.getLogger("chatterbox.training")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune the T3 model.")
    parser.add_argument("config", type=Path, help="Path to the fine-tuning config (YAML or JSON).")
    parser.add_argument("--overrides", nargs="*", default=[], help="Config overrides (key=value).")
    parser.add_argument("--resume", type=Path, default=None, help="Resume from checkpoint file.")
    parser.add_argument("--log-file", type=Path, default=None, help="Optional path to a log file.")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and run evaluation.")
    parser.add_argument("--eval-prompts", type=Path, default=None, help="Optional JSON prompt spec for qualitative eval.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Training device.")
    parser.add_argument("--no-validation", action="store_true", help="Disable validation evaluation.")
    parser.add_argument("--amp", action="store_true", help="Force AMP even if config disables mixed precision.")
    return parser


def _load_and_prepare_config(args: argparse.Namespace) -> T3FineTuningConfig:
    config = load_config(args.config)
    if args.overrides:
        config = apply_overrides(config, args.overrides)
    config.validate()
    return config


def _prepare_trainer(config: T3FineTuningConfig, device: torch.device, *, resume_ckpt: Optional[Path]) -> Trainer:
    model = load_multilingual_t3(config.model, device=device)
    optimizer = build_optimizer(model, config.optimizer)

    total_steps = None
    if config.training.epochs and config.dataset.train_tokens_dir.exists():
        audio_root = config.dataset.audio_root
        if audio_root is not None:
            candidate = (audio_root / config.dataset.train_tokens_dir.name).resolve()
            train_dataset_root = candidate if candidate.exists() else audio_root
        else:
            train_dataset_root = config.dataset.train_tokens_dir.parent
        dataset = T3TokenDataset(
            config.dataset.train_tokens_dir,
            dataset_root=train_dataset_root,
            max_text_len=config.dataset.max_source_tokens,
            max_speech_len=config.dataset.max_target_tokens,
            drop_missing_text=True,
            start_text_token=model.hp.start_text_token,
            stop_text_token=model.hp.stop_text_token,
            start_speech_token=model.hp.start_speech_token,
            stop_speech_token=model.hp.stop_speech_token,
        )
        batches_per_epoch = max(1, -(-len(dataset) // config.dataset.batch_size))  # ceil division
        steps_per_epoch = max(1, -(-batches_per_epoch // config.training.gradient_accumulation_steps))
        total_steps = steps_per_epoch * config.training.epochs
        logger.info("Estimated total steps: %s", total_steps)

    scheduler = build_scheduler(optimizer, config.scheduler, total_steps=total_steps)
    use_amp = bool(config.training.mixed_precision)
    scaler = create_grad_scaler(use_amp)

    trainer = Trainer(
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
    )

    if resume_ckpt is not None:
        with timed_block(f"Loading checkpoint {resume_ckpt}"):
            trainer.load_checkpoint(resume_ckpt)
    return trainer


def _maybe_run_evaluation(trainer: Trainer, config: T3FineTuningConfig, device: torch.device, skip_validation: bool) -> None:
    if skip_validation or trainer.valid_loader is None:
        logger.info("Validation evaluation skipped.")
        return

    metrics = evaluate_losses(
        trainer.model,
        trainer.valid_loader,
        device=device,
        use_amp=trainer.use_amp,
    )
    logger.info(
        "Validation losses: total=%.4f text=%.4f speech=%.4f (ppl text=%.2f, speech=%.2f)",
        metrics["loss_total"],
        metrics["loss_text"],
        metrics["loss_speech"],
        metrics["ppl_text"],
        metrics["ppl_speech"],
    )


def _load_prompt_specs(path: Path, *, base_dir: Optional[Path] = None) -> list[PromptSpec]:
    base = base_dir or path.parent
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Prompt specification must be a list of objects.")

    specs: list[PromptSpec] = []
    for entry in payload:
        if not isinstance(entry, dict):
            raise TypeError("Prompt entries must be mappings.")
        prompt_id = entry.get("prompt_id")
        if not prompt_id:
            raise ValueError("Each prompt entry requires 'prompt_id'.")

        text_tokens = _load_tensor_field(entry.get("text_tokens"), base, dtype=torch.long)
        if text_tokens is None:
            raise ValueError(f"Prompt '{prompt_id}' missing 'text_tokens'.")
        text_tokens = text_tokens.to(dtype=torch.long)

        speaker_embedding = _load_tensor_field(entry.get("speaker_embedding"), base, dtype=torch.float32)

        ref_dict = entry.get("reference_dict")
        if isinstance(ref_dict, str):
            ref_path = _resolve_relative(ref_dict, base)
            ref_obj = torch.load(ref_path, map_location="cpu")
            if not isinstance(ref_obj, dict):
                raise TypeError(f"Reference dict at {ref_path} must decode to a mapping.")
            reference_dict = ref_obj
        else:
            reference_dict = ref_dict if isinstance(ref_dict, dict) else None

        reference_audio = entry.get("reference_audio")
        reference_audio_path = _resolve_relative(reference_audio, base) if reference_audio else None

        specs.append(
            PromptSpec(
                prompt_id=str(prompt_id),
                text_tokens=text_tokens,
                text=entry.get("text"),
                speaker_embedding=speaker_embedding,
                reference_audio=reference_audio_path,
                reference_sr=entry.get("reference_sr"),
                reference_dict=reference_dict,
                metadata=entry.get("metadata"),
                language_id=entry.get("language_id"),
            )
        )
    return specs


def _load_tensor_field(value: Any, base: Path, *, dtype: torch.dtype) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.to(dtype=dtype)
    if isinstance(value, (list, tuple)):
        return torch.tensor(value, dtype=dtype)
    if isinstance(value, str):
        path = _resolve_relative(value, base)
        suffix = path.suffix.lower()
        if suffix in {".pt", ".pth"}:
            obj = torch.load(path, map_location="cpu")
            if isinstance(obj, dict):
                for key in ("text_tokens", "speech_tokens", "tensor", "data"):
                    if key in obj and torch.is_tensor(obj[key]):
                        obj = obj[key]
                        break
            if not torch.is_tensor(obj):
                raise TypeError(f"Tensor file {path} did not contain a tensor.")
            return obj.to(dtype=dtype)
        if suffix == ".npy":
            arr = np.load(path)
            return torch.from_numpy(arr).to(dtype=dtype)
        raise ValueError(f"Unsupported tensor file extension for {path}")
    if isinstance(value, (int, float)):
        return torch.tensor([value], dtype=dtype)
    raise TypeError(f"Unsupported tensor specification: {value!r}")


def _resolve_relative(path: str | Path, base: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (base / candidate).resolve()


def _maybe_run_qualitative(
    trainer: Trainer,
    config: T3FineTuningConfig,
    device: torch.device,
    prompt_file: Optional[Path],
) -> None:
    if prompt_file is None:
        return

    specs = _load_prompt_specs(prompt_file)
    if not specs:
        logger.warning("No prompts found in %s", prompt_file)
        return

    s3gen = load_s3gen(
        checkpoint_dir=config.model.base_checkpoint.parent if config.model.base_checkpoint else None,
        device=device,
    )
    qualitative_dir = config.logging.output_dir / "qualitative"
    results = generate_qualitative_samples(
        trainer.model,
        specs,
        qualitative_dir,
        s3gen=s3gen,
        device=device,
    )
    logger.info("Saved %s qualitative samples to %s", len(results), qualitative_dir)

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        logger_root = setup_logging(level=logging.INFO, log_file=args.log_file, logger_name="chatterbox.training")
        logger_root.info("Starting T3 training with config %s", args.config)

        config = _load_and_prepare_config(args)

        if args.amp:
            config.training.mixed_precision = True

        seed_everything(
            config.seed.python,
            numpy_seed=config.seed.numpy,
            torch_seed=config.seed.torch,
        )

        ensure_directory(config.logging.output_dir)

        device = torch.device(args.device)
        trainer = _prepare_trainer(config, device, resume_ckpt=args.resume)

        if args.eval_only:
            _maybe_run_evaluation(trainer, config, device, args.no_validation)
            logger.info("Eval-only run complete.")
            _maybe_run_qualitative(trainer, config, device, args.eval_prompts)
            return 0

        trainer.train()
        _maybe_run_evaluation(trainer, config, device, args.no_validation)

        if trainer.best_eval_loss is not None:
            logger.info("Best validation loss: %.4f", trainer.best_eval_loss)
        else:
            logger.info("Training complete without validation metrics.")

        _maybe_run_qualitative(trainer, config, device, args.eval_prompts)

        return 0

    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Training failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
