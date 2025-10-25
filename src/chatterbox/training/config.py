"""
Configuration utilities for T3 fine-tuning workflows.

The module defines structured configuration objects for datasets, models,
optimization, training, logging, and seeding. It also provides helpers for
loading config files, applying CLI overrides, and validating that mandatory
paths exist.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Optional

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None


@dataclass
class DatasetConfig:
    train_tokens_dir: Path
    valid_tokens_dir: Optional[Path] = None
    audio_root: Optional[Path] = None
    batch_size: int = 16
    eval_batch_size: Optional[int] = None
    num_workers: int = 4
    max_source_tokens: Optional[int] = None
    max_target_tokens: Optional[int] = None

    def validate(self) -> None:
        if not self.train_tokens_dir.exists():
            raise FileNotFoundError(f"Train token directory not found: {self.train_tokens_dir}")
        if self.valid_tokens_dir is not None and not self.valid_tokens_dir.exists():
            raise FileNotFoundError(f"Validation token directory not found: {self.valid_tokens_dir}")
        if self.audio_root is not None and not self.audio_root.exists():
            raise FileNotFoundError(f"Audio root directory not found: {self.audio_root}")


@dataclass
class ModelConfig:
    base_checkpoint: Optional[Path] = None
    freeze_encoder: bool = False
    freeze_decoder: bool = False
    freeze_modules: tuple[str, ...] = field(default_factory=tuple)

    def validate(self) -> None:
        if self.base_checkpoint is not None and not self.base_checkpoint.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.base_checkpoint}")


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 5e-5
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0


@dataclass
class SchedulerConfig:
    name: str = "linear"
    warmup_steps: int = 0
    min_lr: Optional[float] = None


@dataclass
class TrainingConfig:
    epochs: int = 3
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = True
    max_grad_norm: Optional[float] = None
    eval_every_n_steps: int = 1000


@dataclass
class LoggingConfig:
    output_dir: Path = Path("runs")
    log_every_n_steps: int = 100
    tensorboard_enabled: bool = True
    wandb_enabled: bool = False
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    checkpoint_every_n_steps: int = 1000
    max_checkpoints: Optional[int] = None

    def validate(self) -> None:
        if not str(self.output_dir):
            raise ValueError("Logging output directory must be provided.")
        # No filesystem mutations here; validation only.


@dataclass
class SeedConfig:
    python: int = 1337
    numpy: Optional[int] = None
    torch: Optional[int] = None


@dataclass
class T3FineTuningConfig:
    dataset: DatasetConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    seed: SeedConfig = field(default_factory=SeedConfig)

    def validate(self) -> None:
        self.dataset.validate()
        self.model.validate()
        self.logging.validate()

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "T3FineTuningConfig":
        if "dataset" not in payload:
            raise ValueError("Config requires a 'dataset' section.")

        dataset = _dataset_from_dict(payload["dataset"])
        model = _model_from_dict(payload.get("model", {}))
        optimizer = _optimizer_from_dict(payload.get("optimizer", {}))
        scheduler = _scheduler_from_dict(payload.get("scheduler", {}))
        training = _training_from_dict(payload.get("training", {}))
        logging_cfg = _logging_from_dict(payload.get("logging", {}))
        seed = _seed_from_dict(payload.get("seed", {}))

        return cls(
            dataset=dataset,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            training=training,
            logging=logging_cfg,
            seed=seed,
        )


def load_config(path: str | Path) -> T3FineTuningConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    suffix = config_path.suffix.lower()
    text = config_path.read_text(encoding="utf-8")

    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to load YAML configs.")  # pragma: no cover - depends on optional dep
        payload = yaml.safe_load(text) or {}
    elif suffix == ".json":
        payload = json.loads(text or "{}")
    else:
        raise ValueError(f"Unsupported config extension: {suffix}")

    if not isinstance(payload, Mapping):
        raise TypeError("Top-level config must be a mapping.")

    return T3FineTuningConfig.from_dict(payload)


def apply_overrides(
    config: T3FineTuningConfig,
    overrides: Iterable[str],
) -> T3FineTuningConfig:
    overrides = list(overrides)
    if not overrides:
        return config

    merged: MutableMapping[str, Any] = config.to_dict()
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Overrides must be in key=value form: {override}")
        key, value = override.split("=", 1)
        if not key:
            raise ValueError(f"Override key cannot be empty: {override}")

        path = key.split(".")
        cursor: MutableMapping[str, Any] = merged
        for part in path[:-1]:
            node = cursor.get(part)
            if node is None:
                node = {}
                cursor[part] = node
            elif not isinstance(node, MutableMapping):
                raise ValueError(f"Cannot override '{key}'; '{part}' is not a mapping.")
            cursor = node

        cursor[path[-1]] = _parse_override_value(value)

    return T3FineTuningConfig.from_dict(merged)


def _dataset_from_dict(payload: Mapping[str, Any]) -> DatasetConfig:
    data = dict(payload)
    for key in ("train_tokens_dir", "valid_tokens_dir", "audio_root"):
        if key in data and data[key] is not None:
            data[key] = Path(data[key])
    return DatasetConfig(**data)


def _model_from_dict(payload: Mapping[str, Any]) -> ModelConfig:
    data = dict(payload)
    if "base_checkpoint" in data and data["base_checkpoint"] is not None:
        data["base_checkpoint"] = Path(data["base_checkpoint"])
    if "freeze_modules" in data and data["freeze_modules"] is not None:
        data["freeze_modules"] = tuple(data["freeze_modules"])
    return ModelConfig(**data)


def _optimizer_from_dict(payload: Mapping[str, Any]) -> OptimizerConfig:
    data = dict(payload)
    if "betas" in data and data["betas"] is not None:
        betas = data["betas"]
        if isinstance(betas, (list, tuple)) and len(betas) == 2:
            data["betas"] = (float(betas[0]), float(betas[1]))
        else:
            raise ValueError("Optimizer betas must be a sequence of length 2.")
    return OptimizerConfig(**data)


def _scheduler_from_dict(payload: Mapping[str, Any]) -> SchedulerConfig:
    return SchedulerConfig(**payload)


def _training_from_dict(payload: Mapping[str, Any]) -> TrainingConfig:
    return TrainingConfig(**payload)


def _logging_from_dict(payload: Mapping[str, Any]) -> LoggingConfig:
    data = dict(payload)
    if "output_dir" in data and data["output_dir"] is not None:
        data["output_dir"] = Path(data["output_dir"])
    return LoggingConfig(**data)


def _seed_from_dict(payload: Mapping[str, Any]) -> SeedConfig:
    return SeedConfig(**payload)


def _parse_override_value(value: str) -> Any:
    stripped = value.strip()
    if not stripped:
        return ""

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        lower = stripped.lower()
        if lower == "true":
            return True
        if lower == "false":
            return False
        if lower == "null" or lower == "none":
            return None
        return stripped


def _dataclass_to_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: _dataclass_to_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, Mapping):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, tuple):
        return [_dataclass_to_dict(v) for v in obj]
    if isinstance(obj, list):
        return [_dataclass_to_dict(v) for v in obj]
    return obj


__all__ = [
    "DatasetConfig",
    "ModelConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "TrainingConfig",
    "LoggingConfig",
    "SeedConfig",
    "T3FineTuningConfig",
    "load_config",
    "apply_overrides",
]
