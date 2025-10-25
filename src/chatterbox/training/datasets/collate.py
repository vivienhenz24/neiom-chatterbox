"""
Collation helpers for turning token dataset samples into training batches.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence

from ...models.t3.modules.cond_enc import T3Cond


SpeakerEmbeddingGetter = Callable[[Mapping[str, Any]], Optional[torch.Tensor]]


@dataclass
class T3CollateConfig:
    """
    Configuration controls for :func:`collate_t3`.

    Attributes
    ----------
    speech_pad_value:
        Value used to right-pad speech token sequences.
    text_pad_value:
        Value used to right-pad text token sequences.
    speaker_pad_value:
        Fill value for placeholder speaker embeddings when none are provided in the
        sample metadata.
    speaker_embed_dim:
        Dimensionality of the speaker embedding expected by the model.
    speaker_embedding_getter:
        Optional callable used to extract a speaker embedding tensor from each sample.
        When omitted, the collate function searches common metadata keys.
    create_attention_masks:
        When ``True``, include boolean masks derived from the padded lengths.
    """

    speech_pad_value: int = 0
    text_pad_value: int = 0
    speaker_pad_value: float = 0.0
    speaker_embed_dim: int = 256
    speaker_embedding_getter: Optional[SpeakerEmbeddingGetter] = None
    create_attention_masks: bool = True


def collate_t3(
    batch: Sequence[Mapping[str, Any]],
    config: Optional[T3CollateConfig] = None,
) -> dict[str, Any]:
    """
    Collate a batch of samples produced by :class:`T3TokenDataset`.

    Parameters
    ----------
    batch:
        Sequence of dataset samples (dictionaries with speech/text tokens and metadata).
    config:
        Optional configuration overriding padding values and speaker embedding handling.

    Returns
    -------
    dict
        Dictionary ready for consumption by the T3 trainer, containing padded token
        tensors, lengths, attention masks (optional), metadata fields, and a :class:`T3Cond`
        instance populated with speaker embeddings.
    """
    if not batch:
        raise ValueError("collate_t3 received an empty batch.")

    cfg = config or T3CollateConfig()

    speech_seqs: list[torch.Tensor] = []
    speech_lens: list[int] = []

    text_seqs: list[torch.Tensor] = []
    text_lens: list[int] = []
    missing_text_indices: list[int] = []

    sample_ids: list[str] = []
    audio_paths: list[Optional[str]] = []
    language_ids: list[Optional[str]] = []
    metadata_list: list[Optional[Mapping[str, Any]]] = []
    speaker_embs: list[torch.Tensor] = []
    emotion_adv_values: list[torch.Tensor] = []

    for idx, sample in enumerate(batch):
        sample_ids.append(sample.get("id"))

        speech_tokens = sample.get("speech_tokens")
        if not torch.is_tensor(speech_tokens):
            raise TypeError(f"Sample at index {idx} is missing 'speech_tokens' tensor.")
        speech_tensor = speech_tokens.to(dtype=torch.long, device="cpu").detach()
        speech_len = int(sample.get("speech_token_len", speech_tensor.numel()))

        speech_seqs.append(speech_tensor)
        speech_lens.append(speech_len)

        text_tokens = sample.get("text_tokens")
        if text_tokens is None:
            missing_text_indices.append(idx)
            text_tensor = torch.empty(0, dtype=torch.long)
            text_len = 0
        else:
            if not torch.is_tensor(text_tokens):
                raise TypeError(f"Sample at index {idx} has non-tensor 'text_tokens'.")
            text_tensor = text_tokens.to(dtype=torch.long, device="cpu").detach()
            text_len = int(sample.get("text_token_len", text_tensor.numel()))
        text_seqs.append(text_tensor)
        text_lens.append(text_len)

        audio_path = sample.get("audio_path")
        audio_paths.append(str(audio_path) if audio_path is not None else None)

        language_id = sample.get("language_id")
        language_ids.append(language_id.lower() if isinstance(language_id, str) else None)

        metadata = sample.get("metadata")
        metadata_list.append(metadata if isinstance(metadata, Mapping) else None)

        speaker_tensor = _resolve_speaker_embedding(sample, cfg)
        speaker_embs.append(speaker_tensor)

        emotion_value = _resolve_emotion_adv(sample)
        emotion_adv_values.append(emotion_value)

    if missing_text_indices:
        raise ValueError(
            "One or more samples are missing text tokens. Either regenerate the dataset "
            "with text tokens or filter such samples before collation. Missing indices: "
            f"{missing_text_indices}"
        )

    speech_tokens = pad_sequence(
        speech_seqs,
        batch_first=True,
        padding_value=cfg.speech_pad_value,
    )
    text_tokens = pad_sequence(
        text_seqs,
        batch_first=True,
        padding_value=cfg.text_pad_value,
    )

    speech_lengths = torch.tensor(speech_lens, dtype=torch.long)
    text_lengths = torch.tensor(text_lens, dtype=torch.long)

    speech_mask: Optional[torch.Tensor] = None
    text_mask: Optional[torch.Tensor] = None
    if cfg.create_attention_masks:
        speech_mask = _lengths_to_mask(speech_lengths, speech_tokens.size(1))
        text_mask = _lengths_to_mask(text_lengths, text_tokens.size(1))

    cond = T3Cond(
        speaker_emb=torch.stack(speaker_embs, dim=0),
        emotion_adv=torch.stack(emotion_adv_values, dim=0),
    )

    return {
        "ids": sample_ids,
        "speech_tokens": speech_tokens,
        "speech_token_lens": speech_lengths,
        "speech_attention_mask": speech_mask,
        "text_tokens": text_tokens,
        "text_token_lens": text_lengths,
        "text_attention_mask": text_mask,
        "audio_paths": audio_paths,
        "language_ids": language_ids,
        "metadata": metadata_list,
        "cond": cond,
    }


def _resolve_speaker_embedding(sample: Mapping[str, Any], config: T3CollateConfig) -> torch.Tensor:
    getter = config.speaker_embedding_getter or _default_speaker_embedding_getter
    candidate = getter(sample)

    if candidate is None:
        return torch.full(
            (config.speaker_embed_dim,),
            fill_value=config.speaker_pad_value,
            dtype=torch.float32,
        )

    tensor = candidate.detach().to(dtype=torch.float32, device="cpu")
    if tensor.ndim == 2 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim != 1:
        raise ValueError(
            "Speaker embedding tensors must be 1D (or shape [1, D]). "
            f"Received tensor with shape {tuple(tensor.shape)}."
        )

    if tensor.numel() == config.speaker_embed_dim:
        return tensor

    if tensor.numel() > config.speaker_embed_dim:
        return tensor[: config.speaker_embed_dim]

    out = torch.full(
        (config.speaker_embed_dim,),
        fill_value=config.speaker_pad_value,
        dtype=torch.float32,
    )
    out[: tensor.numel()] = tensor
    return out


def _resolve_emotion_adv(sample: Mapping[str, Any]) -> torch.Tensor:
    metadata = sample.get("metadata")
    value = None
    if isinstance(metadata, Mapping):
        value = metadata.get("emotion_adv") or metadata.get("emotion")

    if value is None:
        value = sample.get("emotion_adv")

    if torch.is_tensor(value):
        tensor = value.detach().to(dtype=torch.float32, device="cpu")
        if tensor.ndim == 0:
            tensor = tensor.view(1)
        elif tensor.ndim > 1:
            tensor = tensor.view(-1)
        return tensor

    try:
        numeric = float(value) if value is not None else 0.5
    except (TypeError, ValueError):
        numeric = 0.5

    return torch.tensor([numeric], dtype=torch.float32)


def _default_speaker_embedding_getter(sample: Mapping[str, Any]) -> Optional[torch.Tensor]:
    direct = sample.get("speaker_embedding")
    if torch.is_tensor(direct):
        return direct

    metadata = sample.get("metadata")
    if isinstance(metadata, Mapping):
        for key in ("speaker_embedding", "speaker_emb"):
            candidate = metadata.get(key)
            if torch.is_tensor(candidate):
                return candidate
    return None


def _lengths_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    if max_len == 0:
        return torch.zeros((lengths.size(0), 0), dtype=torch.bool)
    range_row = torch.arange(max_len, device=lengths.device)
    return range_row.unsqueeze(0) < lengths.unsqueeze(1)


__all__ = ["T3CollateConfig", "collate_t3"]
