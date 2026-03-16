import os
import glob
import numpy as np
import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset


class AudioDeepfakeDataset(Dataset):
    """
    Expects a folder structure like:
      root_dir/
        real/*.wav
        fake/*.wav

    Returns:
      x: feature tensor (e.g., mel spectrogram) with shape (1, n_mels, time)
      y: label tensor (0=real, 1=fake)
    """

    def __init__(self, root_dir, feature_extractor, max_len_sec=4, sr=16000, exts=(".wav", ".flac")):
        self.root_dir = root_dir
        self.sr = sr
        self.max_len = int(max_len_sec * sr)
        self.feat = feature_extractor
        self.exts = exts

        self.samples = []
        for label_name, label in [("real", 0), ("fake", 1)]:
            for ext in self.exts:
                pattern = os.path.join(root_dir, label_name, f"*{ext}")
                for f in glob.glob(pattern):
                    self.samples.append((f, label))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No audio files found in {root_dir}. Expected folders: "
                f"{os.path.join(root_dir,'real')} and {os.path.join(root_dir,'fake')} "
                f"with extensions {self.exts}"
            )

    def __len__(self):
        return len(self.samples)

    def _load_audio(self, path: str) -> torch.Tensor:
        """
        Load audio using soundfile (no ffmpeg required), return tensor shape (1, T).
        """
        wav_np, sr = sf.read(path, always_2d=False)

        # wav_np can be (T,) or (T, C)
        if wav_np.ndim == 2:
            wav_np = wav_np.mean(axis=1)  # -> (T,)

        # Ensure float32
        wav_np = wav_np.astype(np.float32, copy=False)

        wav = torch.from_numpy(wav_np).unsqueeze(0)  # (1, T)

        # Resample to target SR if needed
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)

        return wav

    def _trim_or_pad(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Ensure fixed length self.max_len.
        wav is shape (1, T).
        """
        T = wav.shape[1]
        if T > self.max_len:
            wav = wav[:, :self.max_len]
        elif T < self.max_len:
            pad = self.max_len - T
            wav = torch.nn.functional.pad(wav, (0, pad))
        return wav

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        wav = self._load_audio(path)      # (1, T)
        wav = self._trim_or_pad(wav)      # (1, max_len)

        # Feature extraction (e.g., mel spectrogram)
        x = self.feat(wav)  # typically (1, n_mels, time)

        y = torch.tensor(label, dtype=torch.long)
        return x, y
