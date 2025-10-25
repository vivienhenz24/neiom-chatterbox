"""Generate Luxembourgish samples using a fine-tuned checkpoint and reference voice."""

import argparse
from pathlib import Path

import torch
import torchaudio as ta

from chatterbox.mtl_tts import ChatterboxMultilingualTTS


def load_finetuned_model(device: str, checkpoint: Path) -> ChatterboxMultilingualTTS:
    base_dir = Path("models/multilingual")
    if not base_dir.exists():
        raise FileNotFoundError(
            f"Base multilingual assets not found at {base_dir}. "
            "Run download_multilingual_model.py first."
        )

    model = ChatterboxMultilingualTTS.from_local(base_dir, device=device)

    state = torch.load(checkpoint, map_location="cpu")
    model.t3.load_state_dict(state["model"], strict=True)
    model.t3.to(device).eval()

    return model


def synthesize(
    model: ChatterboxMultilingualTTS,
    output_dir: Path,
    reference_wav: Path,
    lines: dict[str, str],
    cfg_weight: float,
    exaggeration: float,
    temperature: float,
    repetition_penalty: float,
    min_p: float,
    top_p: float,
    
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    model.prepare_conditionals(reference_wav, exaggeration=exaggeration)

    for name, text in lines.items():
        wav = model.generate(
            text,
            language_id="lb",
            audio_prompt_path=reference_wav,
            cfg_weight=cfg_weight,
            exaggeration=exaggeration,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            top_p=top_p,
          
        )
        out_path = output_dir / f"{name}.wav"
        ta.save(str(out_path), wav, model.sr)
        print(f"Wrote {out_path} (length={wav.shape[-1] / model.sr:.2f}s)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Luxembourgish TTS samples")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/runpod_luxembourgish/checkpoints/step_00048000.pt"),
        help="Path to fine-tuned T3 checkpoint to load (defaults to step_00048000.pt).",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Path to a Luxembourgish reference WAV used for conditioning.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs_lux"),
        help="Directory where generated WAVs will be saved.",
    )
    parser.add_argument(
        "--cfg-weight",
        type=float,
        default=0.15,
        help="Classifier-free guidance weight (lower stabilises accent).",
    )
    parser.add_argument(
        "--exaggeration",
        type=float,
        default=0.3,
        help="Emotion/exaggeration factor for T3 conditionals.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for the decoder.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="Penalty to discourage token loops.",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=0.1,
        help="Minimum probability mass for nucleus sampling.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling cutoff.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_finetuned_model(device=device, checkpoint=args.checkpoint)

    lines = {
        "bomi": "Meng bomi huet fréier alt emol en zigarillo gefëmmt.",
        "nottar": "D'nottär huet haut de mueren zwou venten an eng successioun ze traitéieren.",
    }

    synthesize(
        model,
        output_dir=args.output_dir,
        reference_wav=args.reference,
        lines=lines,
        cfg_weight=args.cfg_weight,
        exaggeration=args.exaggeration,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        min_p=args.min_p,
        top_p=args.top_p,
  
    )


if __name__ == "__main__":
    main()
