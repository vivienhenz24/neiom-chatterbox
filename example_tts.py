from pathlib import Path

import torchaudio as ta
import torch

from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Automatically detect the best available device, preferring Apple M-series GPU (MPS) when present.
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# torch APIs in this repo expect strings in some places, so keep both representations handy.
device_str = device.type

print(f"Using device: {device_str}")

print("Loading English TTS model...", flush=True)
model = ChatterboxTTS.from_pretrained(device=device_str)
print("English model ready.", flush=True)

text = "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill."
print("Generating English sample...", flush=True)
wav = model.generate(text)
ta.save("test-1.wav", wav, model.sr)
print("Saved English output to test-1.wav", flush=True)

multilingual_ckpt_dir = Path("models/multilingual")
if not multilingual_ckpt_dir.exists():
    raise FileNotFoundError(
        f"Expected multilingual checkpoints in {multilingual_ckpt_dir}, "
        "but the directory was not found. Run download_multilingual_model.py first."
    )

print(f"Loading multilingual model from {multilingual_ckpt_dir}...", flush=True)
multilingual_model = ChatterboxMultilingualTTS.from_local(multilingual_ckpt_dir, device=device_str)
print("Multilingual model ready.", flush=True)
text = "Bonjour, comment ça va? Ceci est le modèle de synthèse vocale multilingue Chatterbox, il prend en charge 24 langues."
print("Generating French sample...", flush=True)
wav = multilingual_model.generate(text, language_id="fr")
ta.save("test-2.wav", wav, multilingual_model.sr)
print("Saved multilingual output to test-2.wav", flush=True)


# If you want to synthesize with a different voice, specify the audio prompt.
AUDIO_PROMPT_PATH = Path("audio (4).wav")
if not AUDIO_PROMPT_PATH.exists():
    raise FileNotFoundError(
        f"Audio prompt not found at {AUDIO_PROMPT_PATH}. Update the path or provide the file."
    )

print(f"Generating English sample with voice cloning from {AUDIO_PROMPT_PATH}...", flush=True)
wav = model.generate(text, audio_prompt_path=str(AUDIO_PROMPT_PATH))
ta.save("test-3.wav", wav, model.sr)
print("Saved stylized output to test-3.wav", flush=True)
