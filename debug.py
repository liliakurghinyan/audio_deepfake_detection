import torch
import soundfile as sf
import numpy as np
import torchaudio
import json

from features import MelFeatureExtractor
from model import SimpleCNN

PATH = "/Users/lilia/Desktop/zzvanq.wav"

def load_audio(path, target_sr=16000, max_len_sec=4):
    data, sr = sf.read(path, always_2d=False)

    if data.ndim == 2:
        data = data.mean(axis=1)

    data = data.astype(np.float32)
    wav = torch.from_numpy(data).unsqueeze(0)

    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    max_len = target_sr * max_len_sec

    if wav.shape[1] > max_len:
        wav = wav[:, :max_len]
    elif wav.shape[1] < max_len:
        wav = torch.nn.functional.pad(wav, (0, max_len - wav.shape[1]))

    return wav

device = "cuda" if torch.cuda.is_available() else "cpu"

with open("results/best_threshold.json", "r") as f:
    threshold = json.load(f)["threshold"]

model = SimpleCNN().to(device)
model.load_state_dict(torch.load("model.pt", map_location=device))
model.eval()

feat = MelFeatureExtractor(sr=16000, n_mels=80)

wav = load_audio(PATH)
x = feat(wav).unsqueeze(0).to(device)

with torch.no_grad():
    logit = model(x).squeeze()
    prob_fake = torch.sigmoid(logit).item()

print("File:", PATH)
print("Fake probability:", prob_fake)
print("Threshold:", threshold)
print("Prediction:", "FAKE" if prob_fake >= threshold else "REAL")