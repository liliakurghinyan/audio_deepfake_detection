import torch
import torchaudio
import soundfile as sf
import numpy as np
import os

from features import MelFeatureExtractor
from model import SimpleCNN


def load_audio(path: str, target_sr: int = 16000) -> torch.Tensor:
    """
    Loads audio WITHOUT torchaudio.load (so no torchcodec/ffmpeg issues).
    Returns tensor shape (1, T) at target_sr.
    """
    wav_np, sr = sf.read(path, always_2d=False)

    # (T, C) -> mono (T,)
    if wav_np.ndim == 2:
        wav_np = wav_np.mean(axis=1)

    wav_np = wav_np.astype(np.float32, copy=False)
    wav = torch.from_numpy(wav_np).unsqueeze(0)  # (1, T)

    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    return wav


def predict(path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    feat = MelFeatureExtractor(sr=16000, n_mels=80)
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load("model.pt", map_location=device))
    model.eval()

    wav = load_audio(path, 16000)

    # feature extractor returns (1, 80, time); model expects (B,1,80,time)
    x = feat(wav).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(x).squeeze()
        prob_fake = torch.sigmoid(logit).item()

    print(f"File: {path}")
    print(f"Fake probability: {prob_fake:.3f}  (0=real, 1=fake)")


if __name__ == "__main__":
    # Change this to any file you want to test:
    # 1) Put a file named sample.wav in the project folder, OR
    # 2) Put full path here
    predict("sample.wav")
    predict("Screen Recording 2026-02-09 at 18.50.53.wav")

