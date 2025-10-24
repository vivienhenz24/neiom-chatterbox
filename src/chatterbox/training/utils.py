"""
Utility helpers shared across the training stack.

This module provides:
  * Deterministic seeding for Python, NumPy, and Torch.
  * Logging setup with console output and optional file sink.
  * Filesystem helpers for resolving paths and ensuring directories exist.
  * Manifest validation against preprocessing outputs.
  * A lightweight timing context manager for profiling code regions.
"""
from __future__ import annotations

import json
import logging
import os
import random
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Optional

try:  # NumPy is optional for some environments.
    import numpy as _np
except ImportError:  # pragma: no cover - optional dependency
    _np = None

try:  # Torch is optional for inference-only setups.
    import torch as _torch
except ImportError:  # pragma: no cover - optional dependency
    _torch = None


LOGGER = logging.getLogger(__name__)
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]


def seed_everything(
    python_seed: int,
    numpy_seed: Optional[int] = None,
    torch_seed: Optional[int] = None,
) -> None:
    """
    Make common random number generators deterministic.

    Args:
        python_seed: Seed for the Python ``random`` module and PYTHONHASHSEED.
        numpy_seed: Optional override for NumPy's RNG. Defaults to ``python_seed``.
        torch_seed: Optional override for Torch's RNG. Defaults to ``python_seed``.
    """
    LOGGER.debug("Setting deterministic seeds (python=%s, numpy=%s, torch=%s)", python_seed, numpy_seed, torch_seed)
    os.environ["PYTHONHASHSEED"] = str(python_seed)
    random.seed(python_seed)

    if _np is not None:
        _np.random.seed(python_seed if numpy_seed is None else numpy_seed)
    elif numpy_seed is not None:
        LOGGER.warning("NumPy not available; cannot set numpy seed.")

    if _torch is not None:
        actual_seed = python_seed if torch_seed is None else torch_seed
        _torch.manual_seed(actual_seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(actual_seed)
        try:
            _torch.use_deterministic_algorithms(True, warn_only=True)
        except (AttributeError, RuntimeError):  # pragma: no cover - depends on torch version
            LOGGER.debug("Torch.use_deterministic_algorithms not available.")
        cudnn = getattr(_torch.backends, "cudnn", None)
        if cudnn is not None:
            cudnn.benchmark = False
            cudnn.deterministic = True
    elif torch_seed is not None:
        LOGGER.warning("Torch not available; cannot set torch seed.")


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str | Path] = None,
    logger_name: str = "",
) -> logging.Logger:
    """
    Configure logging handlers for the application.

    Args:
        level: Logging level for both console and file handlers.
        log_file: Optional path to a log file. When provided, the directory is created.
        logger_name: Logger name to configure. Defaults to the root logger.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.handlers = []
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        log_path = resolve_path(log_file)
        ensure_directory(log_path.parent)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    LOGGER.debug("Logging configured for logger '%s' (level=%s, file=%s)", logger_name, level, log_file)
    return logger


def resolve_path(path: str | Path, base: Optional[str | Path] = None) -> Path:
    """
    Resolve a path relative to the workspace root (or a provided base).

    Args:
        path: Path to resolve.
        base: Optional base directory. Defaults to ``WORKSPACE_ROOT``.
    """
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    base_path = Path(base).expanduser() if base is not None else WORKSPACE_ROOT
    return (base_path / p).resolve(strict=False)


def ensure_directory(path: str | Path, base: Optional[str | Path] = None) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path to create.
        base: Optional base directory when resolving relative paths.
    """
    directory = resolve_path(path, base=base)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def validate_manifest(
    manifest_path: str | Path,
    expected_splits: Optional[Iterable[str]] = None,
    preprocessing_outputs: Optional[Mapping[str, Any]] = None,
    workspace_root: Optional[str | Path] = None,
) -> Mapping[str, Any]:
    """
    Validate a manifest produced by the preprocessing pipeline.

    Args:
        manifest_path: Path to the manifest JSON file.
        expected_splits: Optional iterable of split names that must be present.
        preprocessing_outputs: Optional mapping of split names to preprocessing
            results (e.g. DataPreparationResult). The function checks that basic
            metrics match the manifest.
        workspace_root: Optional override for resolving relative output dirs.

    Returns:
        Parsed manifest as an immutable mapping.
    """
    resolved_manifest = resolve_path(manifest_path, base=workspace_root)
    if not resolved_manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {resolved_manifest}")

    data = json.loads(resolved_manifest.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("Manifest must decode to a mapping object.")

    splits = data.get("splits")
    if not isinstance(splits, MutableMapping):
        raise ValueError("Manifest missing 'splits' mapping.")

    if expected_splits is not None:
        missing = set(expected_splits) - set(splits)
        if missing:
            raise ValueError(f"Manifest missing expected splits: {sorted(missing)}")

    base_dir = Path(workspace_root).expanduser() if workspace_root is not None else WORKSPACE_ROOT

    for split_name, split_data in splits.items():
        if not isinstance(split_data, MutableMapping):
            raise ValueError(f"Manifest split '{split_name}' must be a mapping.")
        output_dir = split_data.get("output_dir")
        if output_dir is None:
            raise ValueError(f"Manifest split '{split_name}' missing 'output_dir'.")
        output_path = resolve_path(output_dir, base=base_dir)
        if not output_path.exists():
            raise FileNotFoundError(f"Manifest output_dir for split '{split_name}' does not exist: {output_path}")
        split_data["output_dir"] = str(output_path)

        if preprocessing_outputs is not None and split_name in preprocessing_outputs:
            reference = preprocessing_outputs[split_name]
            reference_dict = _as_mapping(reference)
            for key in ("processed", "total_rows", "skipped_audio", "skipped_text"):
                if key in reference_dict and key in split_data and reference_dict[key] != split_data[key]:
                    raise ValueError(
                        f"Manifest mismatch for split '{split_name}' key '{key}': "
                        f"{split_data[key]} != {reference_dict[key]}"
                    )

    return data


def _as_mapping(value: Any) -> Mapping[str, Any]:
    """Convert dataclass or mapping objects into a plain dictionary."""
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    raise TypeError(f"Expected mapping or dataclass, received {type(value)!r}")


@contextmanager
def timed_block(name: str, logger: Optional[logging.Logger] = None, level: int = logging.INFO):
    """
    Measure the execution time of a code block.

    Args:
        name: Human-readable description of the block.
        logger: Optional logger to emit the timing message. Defaults to module logger.
        level: Logging level for the completion message.
    """
    log = logger if logger is not None else LOGGER
    start = time.perf_counter()
    log.debug("Starting timed block: %s", name)
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        log.log(level, "%s completed in %.3f seconds", name, duration)


__all__ = [
    "WORKSPACE_ROOT",
    "ensure_directory",
    "resolve_path",
    "setup_logging",
    "seed_everything",
    "timed_block",
    "validate_manifest",
]
