"""
High-level training loop for T3 fine-tuning.
"""
from __future__ import annotations

import logging
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Protocol, Sequence

import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from .config import T3FineTuningConfig
from .datasets.collate import T3CollateConfig, collate_t3
from .datasets.t3_dataset import T3TokenDataset
from .model_utils import create_grad_scaler

logger = logging.getLogger(__name__)


@dataclass
class TrainStepMetrics:
    global_step: int
    epoch: int
    epoch_step: int
    loss_total: float
    loss_text: float
    loss_speech: float
    learning_rate: float


@dataclass
class EvalMetrics:
    global_step: int
    epoch: int
    loss_total: float
    loss_text: float
    loss_speech: float


class TrainerCallback(Protocol):
    def on_train_step(self, trainer: "Trainer", metrics: TrainStepMetrics) -> None:  # pragma: no cover - optional hook
        ...

    def on_eval_end(self, trainer: "Trainer", metrics: EvalMetrics) -> None:  # pragma: no cover - optional hook
        ...

    def on_epoch_end(self, trainer: "Trainer", epoch: int) -> None:  # pragma: no cover - optional hook
        ...

    def on_save_checkpoint(self, trainer: "Trainer", path: Path) -> None:  # pragma: no cover - optional hook
        ...


class Trainer:
    """
    Coordinates dataset creation, optimization, evaluation, and checkpointing for T3 fine-tuning.
    """

    def __init__(
        self,
        config: T3FineTuningConfig,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        device: torch.device | str = "cpu",
        callbacks: Optional[Sequence[TrainerCallback]] = None,
        eval_fn: Optional[Callable[["Trainer"], dict[str, float]]] = None,
    ) -> None:
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device(device)
        self.callbacks = list(callbacks or [])
        self.eval_fn = eval_fn

        self.grad_accum_steps = max(1, config.training.gradient_accumulation_steps)
        self.use_amp = bool(config.training.mixed_precision and torch.cuda.is_available() and self.device.type == "cuda")
        self.scaler = scaler if scaler is not None else create_grad_scaler(self.use_amp)
        if self.use_amp and self.scaler is None:
            logger.warning("Mixed precision requested but GradScaler unavailable. Falling back to FP32 training.")
            self.use_amp = False

        self.model.to(self.device)

        self.train_loader: DataLoader = self._build_dataloader(
            tokens_dir=config.dataset.train_tokens_dir,
            batch_size=config.dataset.batch_size,
            shuffle=True,
        )
        self.valid_loader: Optional[DataLoader] = None
        if config.dataset.valid_tokens_dir is not None:
            self.valid_loader = self._build_dataloader(
                tokens_dir=config.dataset.valid_tokens_dir,
                batch_size=config.dataset.eval_batch_size or config.dataset.batch_size,
                shuffle=False,
            )

        self.output_dir = config.logging.output_dir.expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._saved_checkpoints: deque[Path] = deque()

        self.global_step = 0
        self.current_epoch = 0
        self.best_eval_loss: Optional[float] = None
        self.should_stop = False

        self.optimizer.zero_grad(set_to_none=True)

    # --------------------------------------------------------------------- #
    # Public API                                                            #
    # --------------------------------------------------------------------- #
    def train(self) -> None:
        epochs = max(1, self.config.training.epochs)
        log_every = max(1, self.config.logging.log_every_n_steps)
        eval_every = max(0, self.config.training.eval_every_n_steps)
        checkpoint_every = max(0, self.config.logging.checkpoint_every_n_steps)

        amp_dtype = torch.float16 if self.use_amp else None
        device_type = self.device.type if self.use_amp else "cpu"

        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            self.model.train()
            running_loss = 0.0
            running_text_loss = 0.0
            running_speech_loss = 0.0
            micro_text_loss = 0.0
            micro_speech_loss = 0.0
            micro_count = 0

            for step, batch in enumerate(self.train_loader):
                batch = self._move_batch_to_device(batch)

                if self.use_amp:
                    autocast_ctx = torch.autocast(device_type=device_type, dtype=amp_dtype)
                else:
                    autocast_ctx = nullcontext()

                with autocast_ctx:
                    text_loss, speech_loss = self.model.loss(
                        t3_cond=batch["cond"],
                        text_tokens=batch["text_tokens"],
                        text_token_lens=batch["text_token_lens"],
                        speech_tokens=batch["speech_tokens"],
                        speech_token_lens=batch["speech_token_lens"],
                    )
                    total_loss = text_loss + speech_loss
                    loss = total_loss / self.grad_accum_steps

                if self.use_amp and self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                micro_text_loss += float(text_loss.detach().cpu())
                micro_speech_loss += float(speech_loss.detach().cpu())
                micro_count += 1

                if self._should_step_optimizer(step):
                    avg_text = micro_text_loss / micro_count
                    avg_speech = micro_speech_loss / micro_count
                    avg_total = avg_text + avg_speech

                    if self.use_amp and self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)

                    if self.config.training.max_grad_norm is not None:
                        clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)

                    if self.use_amp and self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                    if self.scheduler is not None:
                        self.scheduler.step(self.global_step)

                    self.global_step += 1
                    running_loss += avg_total
                    running_text_loss += avg_text
                    running_speech_loss += avg_speech

                    lr = self.optimizer.param_groups[0].get("lr", 0.0)
                    metrics = TrainStepMetrics(
                        global_step=self.global_step,
                        epoch=epoch,
                        epoch_step=step,
                        loss_total=avg_total,
                        loss_text=avg_text,
                        loss_speech=avg_speech,
                        learning_rate=lr,
                    )
                    if self.global_step % log_every == 0:
                        logger.info(
                            "Step %s | Epoch %s | Loss %.4f (text %.4f / speech %.4f) | LR %.3e",
                            self.global_step,
                            epoch,
                            avg_total,
                            avg_text,
                            avg_speech,
                            lr,
                        )

                    self._fire_callbacks("on_train_step", metrics)

                    if checkpoint_every and self.global_step % checkpoint_every == 0:
                        self.save_checkpoint(f"step_{self.global_step:08d}.pt")

                    if eval_every and self.global_step % eval_every == 0 and self.valid_loader is not None:
                        eval_metrics = self.evaluate()
                        self._fire_callbacks("on_eval_end", eval_metrics)

                    if self.should_stop:
                        logger.info("Stopping training loop due to external request.")
                        return

                    micro_text_loss = 0.0
                    micro_speech_loss = 0.0
                    micro_count = 0

            self._fire_callbacks("on_epoch_end", epoch)
            if self.should_stop:
                logger.info("Stopping training loop due to external request.")
                return

        logger.info("Training finished. Total steps: %s", self.global_step)

    def evaluate(self) -> EvalMetrics:
        if self.valid_loader is None:
            raise RuntimeError("No validation dataloader configured.")

        self.model.eval()
        total_loss = 0.0
        total_text = 0.0
        total_speech = 0.0
        batches = 0

        with torch.no_grad():
            for batch in self.valid_loader:
                batch = self._move_batch_to_device(batch)
                text_loss, speech_loss = self.model.loss(
                    t3_cond=batch["cond"],
                    text_tokens=batch["text_tokens"],
                    text_token_lens=batch["text_token_lens"],
                    speech_tokens=batch["speech_tokens"],
                    speech_token_lens=batch["speech_token_lens"],
                )
                total_text += float(text_loss.detach().cpu())
                total_speech += float(speech_loss.detach().cpu())
                total_loss += float((text_loss + speech_loss).detach().cpu())
                batches += 1

        if batches == 0:
            raise RuntimeError("Validation dataloader produced zero batches.")

        metrics = EvalMetrics(
            global_step=self.global_step,
            epoch=self.current_epoch,
            loss_total=total_loss / batches,
            loss_text=total_text / batches,
            loss_speech=total_speech / batches,
        )

        logger.info(
            "Eval @ step %s | Loss %.4f (text %.4f / speech %.4f)",
            metrics.global_step,
            metrics.loss_total,
            metrics.loss_text,
            metrics.loss_speech,
        )

        if self.eval_fn is not None:
            extra_metrics = self.eval_fn(self) or {}
            for key, value in extra_metrics.items():
                logger.info("Eval metric %s: %s", key, value)

        if self.best_eval_loss is None or metrics.loss_total < self.best_eval_loss:
            self.best_eval_loss = metrics.loss_total
            self.save_checkpoint("best.pt")

        self.model.train()
        return metrics

    def save_checkpoint(self, filename: str) -> Path:
        path = self.checkpoint_dir / filename
        state = {
            "global_step": self.global_step,
            "epoch": self.current_epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler": self.scaler.state_dict() if self.scaler is not None and self.use_amp else None,
            "best_eval_loss": self.best_eval_loss,
            "config": self.config.to_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        torch.save(state, path)
        logger.info("Checkpoint saved to %s", path)
        self._track_checkpoint(path)
        self._fire_callbacks("on_save_checkpoint", path)
        return path

    def load_checkpoint(
        self,
        path: Path | str,
        *,
        load_optimizer: bool = True,
        load_scheduler: bool = True,
        load_scaler: bool = True,
    ) -> None:
        ckpt_path = Path(path).expanduser().resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        logger.info("Loading checkpoint from %s", ckpt_path)
        state = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(state["model"])

        if load_optimizer and "optimizer" in state and state["optimizer"] is not None:
            self.optimizer.load_state_dict(state["optimizer"])
        if load_scheduler and self.scheduler is not None and "scheduler" in state and state["scheduler"] is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        if (
            load_scaler
            and self.scaler is not None
            and self.use_amp
            and "scaler" in state
            and state["scaler"] is not None
        ):
            self.scaler.load_state_dict(state["scaler"])

        self.global_step = int(state.get("global_step", 0))
        self.current_epoch = int(state.get("epoch", 0))
        self.best_eval_loss = state.get("best_eval_loss")

        torch_rng = state.get("torch_rng_state")
        if torch_rng is not None:
            torch.set_rng_state(torch_rng)
        if torch.cuda.is_available() and state.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng_state"])

        logger.info(
            "Checkpoint loaded. Resuming from epoch %s, global step %s.",
            self.current_epoch,
            self.global_step,
        )

    def register_callback(self, callback: TrainerCallback) -> None:
        self.callbacks.append(callback)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _build_dataloader(self, tokens_dir: Path, batch_size: int, shuffle: bool) -> DataLoader:
        tokens_dir = tokens_dir.expanduser().resolve()
        if not tokens_dir.exists():
            raise FileNotFoundError(f"Token directory not found: {tokens_dir}")

        audio_root_base = self.config.dataset.audio_root
        if audio_root_base is not None:
            candidate = (audio_root_base / tokens_dir.name).resolve()
            dataset_root = candidate if candidate.exists() else audio_root_base
        else:
            dataset_root = tokens_dir.parent
        dataset = T3TokenDataset(
            tokens_dir,
            dataset_root=dataset_root,
            max_text_len=self.config.dataset.max_source_tokens,
            max_speech_len=self.config.dataset.max_target_tokens,
            drop_missing_text=True,
            start_text_token=self.model.hp.start_text_token,
            stop_text_token=self.model.hp.stop_text_token,
            start_speech_token=self.model.hp.start_speech_token,
            stop_speech_token=self.model.hp.stop_speech_token,
        )

        collate_cfg = T3CollateConfig()
        pin_memory = self.device.type == "cuda"
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config.dataset.num_workers,
            pin_memory=pin_memory,
            persistent_workers=self.config.dataset.num_workers > 0,
            collate_fn=lambda batch: collate_t3(batch, collate_cfg),
        )
        logger.info(
            "Built dataloader for %s (%s samples, batch_size=%s, shuffle=%s)",
            tokens_dir,
            len(dataset),
            batch_size,
            shuffle,
        )
        return loader

    def _move_batch_to_device(self, batch: dict[str, torch.Tensor | list | None]) -> dict[str, torch.Tensor | list | None]:
        cond = batch["cond"]
        if hasattr(cond, "to"):
            cond = cond.to(device=self.device)
            batch["cond"] = cond

        for key in (
            "speech_tokens",
            "speech_token_lens",
            "speech_attention_mask",
            "text_tokens",
            "text_token_lens",
            "text_attention_mask",
        ):
            tensor = batch.get(key)
            if torch.is_tensor(tensor):
                batch[key] = tensor.to(self.device, non_blocking=True)

        return batch

    def _should_step_optimizer(self, dataloader_step: int) -> bool:
        if (dataloader_step + 1) % self.grad_accum_steps == 0:
            return True
        total_batches = len(self.train_loader)
        return dataloader_step + 1 == total_batches

    def _track_checkpoint(self, path: Path) -> None:
        self._saved_checkpoints.append(path)
        max_ckpts = self.config.logging.max_checkpoints
        if max_ckpts is not None:
            while len(self._saved_checkpoints) > max_ckpts:
                old = self._saved_checkpoints.popleft()
                if old.exists():
                    old.unlink()
                    logger.info("Removed old checkpoint %s to maintain retention limit.", old)

    def _fire_callbacks(self, event: str, *args) -> None:
        for callback in self.callbacks:
            handler = getattr(callback, event, None)
            if handler is not None:
                handler(self, *args)


__all__ = ["Trainer", "TrainStepMetrics", "EvalMetrics", "TrainerCallback"]
