"""
Model loading and optimization utilities for T3 fine-tuning.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import torch
from safetensors.torch import load_file as load_safetensors
from torch import nn
from torch.cuda.amp import GradScaler as _CudaGradScaler
from torch.optim import Optimizer
try:  # PyTorch >= 2.1 exposes the new torch.amp API.
    from torch.amp import GradScaler as _AmpGradScaler  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - older torch versions
    _AmpGradScaler = None

GradScaler = _AmpGradScaler or _CudaGradScaler

from ..models.s3gen import S3Gen
from ..models.tokenizers import MTLTokenizer
from ..models.t3.t3 import T3
from ..models.t3.modules.t3_config import T3Config
from ..models.t3.utils import ensure_text_vocab_capacity
from .config import ModelConfig, OptimizerConfig, SchedulerConfig

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_MULTILINGUAL_DIR = _REPO_ROOT / "models" / "multilingual"
_DEFAULT_T3_FILENAME = "t3_mtl23ls_v2.safetensors"
_DEFAULT_S3GEN_FILENAME = "s3gen.pt"


def load_multilingual_t3(
    model_cfg: ModelConfig,
    *,
    device: torch.device | str = "cpu",
    dtype: Optional[torch.dtype] = None,
) -> T3:
    """
    Instantiate a multilingual T3 model, load weights, optionally freeze modules, and move to the target device.

    Parameters
    ----------
    model_cfg:
        Model configuration describing checkpoint location and freezing directives.
    device:
        Target device for the returned model.
    dtype:
        Optional dtype cast applied to model parameters and buffers.
    """
    checkpoint_path = _resolve_t3_checkpoint(model_cfg.base_checkpoint)
    vocab_size = _resolve_tokenizer_vocab_size(checkpoint_path)
    model = T3(T3Config.multilingual(text_tokens_dict_size=vocab_size))
    state_dict = _load_t3_state_dict(checkpoint_path)
    state_dict = ensure_text_vocab_capacity(state_dict, vocab_size)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys when loading T3 state dict: {unexpected}")
    if missing:
        logger.warning("Missing parameters when loading T3 state dict: %s", missing)

    _apply_freezing(
        model,
        freeze_encoder=model_cfg.freeze_encoder,
        freeze_decoder=model_cfg.freeze_decoder,
        extra_modules=model_cfg.freeze_modules,
    )

    model.to(device=device, dtype=dtype)
    model.train()
    return model


def _resolve_t3_checkpoint(provided: Optional[Path]) -> Path:
    if provided is not None:
        path = provided.expanduser()
        if path.is_dir():
            candidate = path / _DEFAULT_T3_FILENAME
            if candidate.exists():
                return candidate
        if path.exists():
            return path
        raise FileNotFoundError(f"T3 checkpoint not found at {path}")

    default_path = _DEFAULT_MULTILINGUAL_DIR / _DEFAULT_T3_FILENAME
    if default_path.exists():
        return default_path
    raise FileNotFoundError(
        "No base checkpoint provided and default multilingual checkpoint missing. "
        f"Expected at {default_path}"
    )


def _resolve_tokenizer_vocab_size(checkpoint_path: Path) -> int:
    directory = checkpoint_path if checkpoint_path.is_dir() else checkpoint_path.parent
    tokenizer_path = directory / "grapheme_mtl_merged_expanded_v1.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer JSON not found next to checkpoint at {tokenizer_path}")

    tokenizer = MTLTokenizer(str(tokenizer_path))
    vocab_size = len(tokenizer.tokenizer.get_vocab())
    logger.info("Detected multilingual tokenizer with %d tokens.", vocab_size)
    return vocab_size


def _load_t3_state_dict(path: Path) -> dict[str, torch.Tensor]:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"T3 checkpoint file not found: {path}")

    if path.suffix == ".safetensors":
        state = load_safetensors(str(path))
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)

    if "state_dict" in state:
        state = state["state_dict"]
    elif "model_state_dict" in state:
        state = state["model_state_dict"]
    elif "model" in state:
        model_state = state["model"]
        if isinstance(model_state, (list, tuple)):
            state = model_state[0]
        else:
            state = model_state

    if not isinstance(state, dict):
        raise TypeError(f"Unsupported state-dict format in checkpoint: {path}")

    return state


def _apply_freezing(
    model: T3,
    *,
    freeze_encoder: bool,
    freeze_decoder: bool,
    extra_modules: Sequence[str],
) -> None:
    modules: list[nn.Module] = []

    if freeze_encoder:
        modules.extend(
            filter(None, [model.tfmr, getattr(model, "text_emb", None), getattr(model, "text_pos_emb", None)])
        )
    if freeze_decoder:
        modules.extend(
            filter(
                None,
                [getattr(model, "speech_emb", None), getattr(model, "speech_pos_emb", None), getattr(model, "speech_head", None)],
            )
        )

    for name in extra_modules:
        module = _get_submodule(model, name)
        if module is None:
            logger.warning("Ignoring unknown module in freeze list: %s", name)
            continue
        modules.append(module)

    seen = set()
    for module in modules:
        if module in seen:
            continue
        for param in module.parameters():
            param.requires_grad = False
        module.eval()
        seen.add(module)


def _get_submodule(root: nn.Module, name: str) -> Optional[nn.Module]:
    cursor: nn.Module = root
    for part in name.split("."):
        if part.isdigit():
            if isinstance(cursor, (nn.ModuleList, nn.Sequential, list, tuple)):
                index = int(part)
                if index >= len(cursor):  # type: ignore[arg-type]
                    return None
                cursor = cursor[index]  # type: ignore[index]
            else:
                return None
        else:
            if not hasattr(cursor, part):
                return None
            cursor = getattr(cursor, part)
        if not isinstance(cursor, nn.Module):
            return None
    return cursor


def build_optimizer(model: nn.Module, opt_cfg: OptimizerConfig) -> Optimizer:
    name = opt_cfg.name.lower()
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable parameters remain after applying freezing directives.")

    if name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=opt_cfg.lr,
            betas=opt_cfg.betas,
            eps=opt_cfg.eps,
            weight_decay=opt_cfg.weight_decay,
        )
    if name == "adam":
        return torch.optim.Adam(
            params,
            lr=opt_cfg.lr,
            betas=opt_cfg.betas,
            eps=opt_cfg.eps,
            weight_decay=opt_cfg.weight_decay,
        )
    if name == "sgd":
        return torch.optim.SGD(params, lr=opt_cfg.lr, weight_decay=opt_cfg.weight_decay, momentum=0.9)

    raise ValueError(f"Unsupported optimizer: {opt_cfg.name}")


def build_scheduler(
    optimizer: Optimizer,
    sched_cfg: SchedulerConfig,
    *,
    total_steps: Optional[int],
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    name = (sched_cfg.name or "none").lower()
    if name in {"none", "constant"} and sched_cfg.warmup_steps == 0:
        return None

    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])

    if name == "none":
        warmup = sched_cfg.warmup_steps
        if warmup <= 0:
            return None
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.0,
            end_factor=1.0,
            total_iters=warmup,
        )

    if total_steps is None or total_steps <= 0:
        raise ValueError("total_steps must be provided for scheduler configuration.")

    warmup = max(0, sched_cfg.warmup_steps)
    min_lr = sched_cfg.min_lr or 0.0
    main_steps = max(1, total_steps - warmup)

    if name == "linear":
        return _linear_decay_scheduler(optimizer, warmup, main_steps, min_lr)

    if name == "cosine":
        schedulers: list[torch.optim.lr_scheduler.LRScheduler] = []
        milestones: list[int] = []
        if warmup:
            schedulers.append(
                torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=0.0,
                    end_factor=1.0,
                    total_iters=warmup,
                )
            )
            milestones.append(warmup)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=main_steps,
            eta_min=min_lr,
        )
        if not schedulers:
            return cosine
        schedulers.append(cosine)
        return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=schedulers, milestones=milestones)

    raise ValueError(f"Unsupported scheduler: {sched_cfg.name}")


def _linear_decay_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    decay_steps: int,
    min_lr: float,
) -> torch.optim.lr_scheduler.LRScheduler:
    min_lrs = []
    for group in optimizer.param_groups:
        base_lr = group.get("initial_lr", group["lr"])
        min_ratio = min_lr / base_lr if base_lr > 0 else 0.0
        min_lrs.append(max(min_ratio, 0.0))

    def make_lambda(min_ratio: float):
        def schedule(step: int) -> float:
            if step < warmup_steps:
                if warmup_steps == 0:
                    return 1.0
                return max(step / max(1, warmup_steps), min_ratio)
            progress = (step - warmup_steps) / max(1, decay_steps)
            scale = 1.0 - progress
            return max(scale, min_ratio)

        return schedule

    lambdas = [make_lambda(ratio) for ratio in min_lrs]
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)


def create_grad_scaler(enable_amp: bool) -> Optional[GradScaler]:
    if not enable_amp:
        return None
    if not torch.cuda.is_available():
        logger.warning("AMP enabled but CUDA not available; returning None for GradScaler.")
        return None
    if _AmpGradScaler is not None:
        try:
            return _AmpGradScaler(device_type="cuda")
        except TypeError:  # Older signatures may not support kwargs
            return _AmpGradScaler("cuda")  # type: ignore[call-arg]
    return _CudaGradScaler()


def load_s3gen(
    *,
    checkpoint_dir: Optional[Path],
    device: torch.device | str = "cpu",
    dtype: Optional[torch.dtype] = None,
) -> Optional[S3Gen]:
    if checkpoint_dir is None:
        default_ckpt = _DEFAULT_MULTILINGUAL_DIR / _DEFAULT_S3GEN_FILENAME
        if not default_ckpt.exists():
            return None
        checkpoint_dir = default_ckpt

    checkpoint_dir = checkpoint_dir.expanduser()
    if checkpoint_dir.is_dir():
        checkpoint_path = checkpoint_dir / _DEFAULT_S3GEN_FILENAME
    else:
        checkpoint_path = checkpoint_dir

    if not checkpoint_path.exists():
        logger.warning("S3Gen checkpoint not found at %s", checkpoint_path)
        return None

    s3gen = S3Gen()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    s3gen.load_state_dict(state)
    s3gen.to(device=device, dtype=dtype).eval()
    return s3gen


__all__ = [
    "build_optimizer",
    "build_scheduler",
    "create_grad_scaler",
    "load_multilingual_t3",
    "load_s3gen",
]
