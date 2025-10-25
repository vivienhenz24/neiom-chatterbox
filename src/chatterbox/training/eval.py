"""
Evaluation utilities for T3 fine-tuning workflows.
"""
from __future__ import annotations

import json
import logging
import math
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import torch

from ..models.s3gen import S3GEN_SR, S3Gen
from ..models.t3.modules.cond_enc import T3Cond
from ..models.t3.t3 import T3

logger = logging.getLogger(__name__)


@dataclass
class PromptSpec:
    """
    Description of a qualitative evaluation prompt.

    Attributes
    ----------
    prompt_id:
        Unique identifier used for filenames and metadata.
    text_tokens:
        Tensor containing BOS/EOS wrapped text token ids.
    text:
        Human-readable transcript (optional, but recommended for metadata).
    speaker_embedding:
        Optional speaker embedding tensor. When omitted, a zero vector is used.
    reference_audio:
        Path to a reference waveform used to drive S3Gen. Ignored if reference_dict
        is provided.
    reference_sr:
        Sample rate of ``reference_audio`` when it cannot be inferred from file.
    reference_dict:
        Optional pre-computed reference bundle compatible with ``S3Gen.inference``.
    metadata:
        Additional metadata persisted alongside generated outputs.
    language_id:
        Optional language identifier forwarded to metadata.
    """

    prompt_id: str
    text_tokens: torch.Tensor
    text: Optional[str] = None
    speaker_embedding: Optional[torch.Tensor] = None
    reference_audio: Optional[Path] = None
    reference_sr: Optional[int] = None
    reference_dict: Optional[dict] = None
    metadata: Optional[dict] = None
    language_id: Optional[str] = None


def evaluate_losses(
    model: torch.nn.Module,
    dataloader: Iterable[dict[str, torch.Tensor]],
    *,
    device: torch.device | str,
    use_amp: bool = False,
) -> dict[str, float]:
    """
    Compute mean validation losses and perplexities over ``dataloader``.

    Parameters
    ----------
    model:
        Trained T3 model (expects ``loss`` method returning text/speech losses).
    dataloader:
        Iterable yielding batches produced by :func:`collate_t3`.
    device:
        Device to run evaluation on.
    use_amp:
        Enable automatic mixed precision during evaluation.
    """
    model_device = torch.device(device)
    model = model.to(model_device)
    was_training = model.training
    model.eval()

    total_text = 0.0
    total_speech = 0.0
    total = 0.0
    batches = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = _move_batch_to_device(batch, model_device)
            if use_amp:
                autocast_ctx = torch.autocast(device_type=model_device.type, dtype=torch.float16)
            else:
                autocast_ctx = nullcontext()
            with autocast_ctx:
                text_loss, speech_loss = model.loss(
                    t3_cond=batch["cond"],
                    text_tokens=batch["text_tokens"],
                    text_token_lens=batch["text_token_lens"],
                    speech_tokens=batch["speech_tokens"],
                    speech_token_lens=batch["speech_token_lens"],
                )

            text_value = float(text_loss.detach().cpu())
            speech_value = float(speech_loss.detach().cpu())
            total_text += text_value
            total_speech += speech_value
            total += text_value + speech_value
            batches += 1

    if was_training:
        model.train()

    if batches == 0:
        raise RuntimeError("Validation dataloader produced zero batches.")

    mean_text = total_text / batches
    mean_speech = total_speech / batches
    mean_total = total / batches

    return {
        "loss_text": mean_text,
        "loss_speech": mean_speech,
        "loss_total": mean_total,
        "ppl_text": math.exp(mean_text) if math.isfinite(mean_text) else float("inf"),
        "ppl_speech": math.exp(mean_speech) if math.isfinite(mean_speech) else float("inf"),
    }


def generate_qualitative_samples(
    model: T3,
    prompts: Sequence[PromptSpec],
    output_dir: Path | str,
    *,
    s3gen: Optional[S3Gen] = None,
    device: torch.device | str = "cpu",
    max_new_tokens: Optional[int] = None,
    do_sample: bool = False,
    temperature: float = 0.8,
    top_p: float = 0.95,
    cfg_weight: float = 0.0,
) -> list[dict[str, object]]:
    """
    Decode a fixed set of prompts through T3 (and optionally S3Gen) for qualitative review.

    Parameters
    ----------
    model:
        Fine-tuned T3 model.
    prompts:
        Iterable of :class:`PromptSpec` entries.
    output_dir:
        Directory where audio files and metadata JSON will be written.
    s3gen:
        Optional S3Gen instance for waveform synthesis. When omitted, only speech tokens
        are stored.
    device:
        Target device for inference.
    max_new_tokens:
        Limit on generated speech tokens. Defaults to ``model.hp.max_speech_tokens``.
    do_sample / temperature / top_p / cfg_weight:
        Sampling parameters forwarded to :meth:`T3.inference`.

    Returns
    -------
    list of dict
        Metadata entries describing each generated sample.
    """
    device_obj = torch.device(device)
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    model = model.to(device_obj)
    was_training = model.training
    model.eval()

    results: list[dict[str, object]] = []
    metadata_records: list[dict[str, object]] = []

    with torch.no_grad():
        for spec in prompts:
            text_tokens = spec.text_tokens.to(device_obj, dtype=torch.long)
            speaker_emb = spec.speaker_embedding
            if speaker_emb is None:
                speaker_emb = torch.zeros(model.hp.speaker_embed_size, dtype=torch.float32)
            speaker_emb = speaker_emb.to(device_obj, dtype=torch.float32)

            cond = T3Cond(speaker_emb=speaker_emb)
            cond = cond.to(device=device_obj)

            generated_tokens = model.inference(
                t3_cond=cond,
                text_tokens=text_tokens,
                max_new_tokens=max_new_tokens or model.hp.max_speech_tokens,
                stop_on_eos=True,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                cfg_weight=cfg_weight,
            )
            generated_tokens = generated_tokens.to("cpu")

            sample_dict: dict[str, object] = {
                "id": spec.prompt_id,
                "text": spec.text,
                "language_id": spec.language_id,
                "num_speech_tokens": int(generated_tokens.shape[-1]),
            }

            token_path = output_path / f"{spec.prompt_id}_tokens.pt"
            torch.save({"speech_tokens": generated_tokens}, token_path)
            sample_dict["token_path"] = str(token_path)

            audio_path: Optional[Path] = None
            if s3gen is not None:
                audio = _synthesize_with_s3gen(
                    s3gen,
                    generated_tokens,
                    device=device_obj,
                    reference_audio=spec.reference_audio,
                    reference_sr=spec.reference_sr,
                    reference_dict=spec.reference_dict,
                )
                if audio is not None:
                    audio_path = output_path / f"{spec.prompt_id}.wav"
                    _save_audio(audio, audio_path, sample_rate=S3GEN_SR)
                    sample_dict["audio_path"] = str(audio_path)

            if spec.metadata:
                sample_dict["metadata"] = spec.metadata

            results.append(sample_dict)
            metadata_records.append(dict(sample_dict))

    if was_training:
        model.train()

    metadata_file = output_path / "metadata.json"
    metadata_file.write_text(json.dumps(metadata_records, indent=2), encoding="utf-8")
    logger.info("Qualitative samples saved to %s (metadata: %s)", output_path, metadata_file)

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _move_batch_to_device(
    batch: dict[str, torch.Tensor | list | None],
    device: torch.device,
) -> dict[str, torch.Tensor | list | None]:
    cond = batch["cond"]
    if hasattr(cond, "to"):
        batch["cond"] = cond.to(device=device)

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
            batch[key] = tensor.to(device, non_blocking=True)

    return batch


def _synthesize_with_s3gen(
    s3gen: S3Gen,
    speech_tokens: torch.Tensor,
    *,
    device: torch.device,
    reference_audio: Optional[Path],
    reference_sr: Optional[int],
    reference_dict: Optional[dict],
) -> Optional[torch.Tensor]:
    if reference_dict is not None:
        ref_dict: dict[str, torch.Tensor | None] = {}
        for key, value in reference_dict.items():
            if torch.is_tensor(value):
                ref_dict[key] = value.to(device)
            elif isinstance(value, np.ndarray):
                ref_dict[key] = torch.from_numpy(value).to(device)
            elif isinstance(value, (list, tuple)):
                ref_dict[key] = torch.as_tensor(value).to(device)
            else:
                ref_dict[key] = value
        ref_wav = None
        ref_sr = None
    elif reference_audio is not None:
        import torchaudio

        wav, sr = torchaudio.load(str(reference_audio))
        if wav.ndim > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if reference_sr is not None:
            sr = reference_sr
        ref_wav = wav.to(device)
        ref_dict = None
        ref_sr = sr
    else:
        logger.warning("No reference audio provided for prompt; skipping waveform synthesis.")
        return None

    tokens = speech_tokens.to(device=device, dtype=torch.long)
    s3gen = s3gen.to(device)

    if ref_dict is not None:
        wav, *_ = s3gen.inference(tokens, ref_dict=ref_dict, finalize=True)
    else:
        wav, *_ = s3gen.inference(tokens, ref_wav=ref_wav, ref_sr=ref_sr, finalize=True)
    return wav.squeeze(0).cpu()


def _save_audio(audio: torch.Tensor, path: Path, *, sample_rate: int) -> None:
    import torchaudio

    waveform = audio.unsqueeze(0)
    torchaudio.save(str(path), waveform, sample_rate)


__all__ = ["PromptSpec", "evaluate_losses", "generate_qualitative_samples"]
