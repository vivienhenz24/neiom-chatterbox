from __future__ import annotations

import logging
from typing import Dict

import torch


logger = logging.getLogger(__name__)


def ensure_text_vocab_capacity(
    state_dict: Dict[str, torch.Tensor],
    target_vocab_size: int,
    init_std: float = 0.02,
) -> Dict[str, torch.Tensor]:
    """
    Ensure that text embedding and projection matrices in a T3 state dict
    have at least ``target_vocab_size`` rows. If the checkpoint was trained
    with a smaller vocabulary, new rows are appended and initialised from a
    normal distribution with ``init_std``.

    Parameters
    ----------
    state_dict:
        Model state dictionary to mutate.
    target_vocab_size:
        Desired vocabulary size (number of rows).
    init_std:
        Standard deviation for the random initialisation of new rows.
    """
    if target_vocab_size is None:
        return state_dict

    if target_vocab_size <= 0:
        raise ValueError("target_vocab_size must be positive.")

    def _expand_matrix(param_name: str) -> None:
        weight = state_dict.get(param_name)
        if weight is None:
            return

        current_rows = weight.shape[0]
        if current_rows >= target_vocab_size:
            return

        rows_to_add = target_vocab_size - current_rows
        logger.info(
            "Expanding %s from %d to %d rows to accommodate new vocabulary tokens.",
            param_name,
            current_rows,
            target_vocab_size,
        )
        new_rows = weight.new_empty((rows_to_add, weight.shape[1])).normal_(mean=0.0, std=init_std)
        state_dict[param_name] = torch.cat([weight, new_rows], dim=0)

    _expand_matrix("text_emb.weight")
    _expand_matrix("text_head.weight")
    return state_dict
