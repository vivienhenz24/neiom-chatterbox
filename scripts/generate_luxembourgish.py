from pathlib import Path

import torch
import torchaudio as ta

from chatterbox.mtl_tts import ChatterboxMultilingualTTS


def load_finetuned_model(device: str) -> ChatterboxMultilingualTTS:
    base_dir = Path("models/multilingual")
    if not base_dir.exists():
        raise FileNotFoundError(
            f"Base multilingual assets not found at {base_dir}. "
            "Run download_multilingual_model.py first."
        )

    model = ChatterboxMultilingualTTS.from_local(base_dir, device=device)

    finetuned_ckpt = Path("runs/runpod_luxembourgish/checkpoints/best.pt")
    if not finetuned_ckpt.exists():
        raise FileNotFoundError(f"Fine-tuned checkpoint not found: {finetuned_ckpt}")

    state = torch.load(finetuned_ckpt, map_location="cpu")
    model.t3.load_state_dict(state["model"], strict=True)
    model.t3.to(device).eval()

    return model


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_finetuned_model(device=device)

    lines = {
        "bomi": "Meng bomi huet fréier alt emol en zigarillo gefëmmt.",
        "nottar": "D'nottär huet haut de mueren zwou venten an eng successioun ze traitéieren.",
    }

    for key, text in lines.items():
        wav = model.generate(text, language_id="lb")
        ta.save(f"{key}.wav", wav, model.sr)
        print(f"Wrote {key}.wav (len={wav.shape[-1]/model.sr:.2f}s)")


if __name__ == "__main__":
    main()
