"""
Lightweight fallback implementation of the perth implicit watermarker.

The original `perth-implicit-watermark` package is optional and not always
available in constrained environments (such as the fine-tuning pods).  We only
need the `PerthImplicitWatermarker.apply_watermark` method during sample
generation, so this stub simply returns the waveform unchanged.
"""

from __future__ import annotations


class PerthImplicitWatermarker:
    """No-op placeholder used when the real perth package is unavailable."""

    def apply_watermark(self, wav, sample_rate: int):  # noqa: ARG002 - interface parity
        return wav

