import os
import tempfile
import numpy as np
import torch
import torchaudio
import soundfile as sf
from pydub import AudioSegment

from features import MelFeatureExtractor
from model import SimpleCNN


class AudioDeepfakeDetector:
    def __init__(self, model_path="model.pt", sr=16000, n_mels=80):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sr = sr
        self.feat = MelFeatureExtractor(sr=sr, n_mels=n_mels)

        self.model = SimpleCNN().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def _convert_file_to_wav_path(self, audio_bytes: bytes, filename: str):
        """
        Save uploaded file temporarily and convert to WAV if needed.
        Returns:
            wav_path, converted_flag, original_ext
        """
        ext = os.path.splitext(filename)[1].lower()
        original_ext = ext.replace(".", "") if ext else "unknown"

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext or ".bin") as tmp_in:
            tmp_in.write(audio_bytes)
            input_path = tmp_in.name

        # If already wav, use as-is
        if ext == ".wav":
            return input_path, False, original_ext

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_out:
            output_path = tmp_out.name

        try:
            # Let pydub/ffmpeg infer format from the file itself
            audio = AudioSegment.from_file(input_path)
            audio.export(output_path, format="wav")
            os.remove(input_path)
            return output_path, True, original_ext
        except Exception as e:
            # Clean temp files if conversion fails
            if os.path.exists(input_path):
                os.remove(input_path)
            if os.path.exists(output_path):
                os.remove(output_path)
            raise RuntimeError(
                f"Could not convert uploaded file '{filename}' to WAV. "
                f"The file may be corrupted or may not match its extension."
            ) from e

    def _load_audio_from_path(self, path: str) -> torch.Tensor:
        data, sr = sf.read(path, always_2d=False)

        if data.ndim == 2:
            data = data.mean(axis=1)

        data = data.astype(np.float32, copy=False)
        wav = torch.from_numpy(data).unsqueeze(0)

        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)

        return wav

    @torch.no_grad()
    def predict_from_upload(self, audio_bytes: bytes, filename: str):
        wav_path, converted, original_ext = self._convert_file_to_wav_path(audio_bytes, filename)

        try:
            wav = self._load_audio_from_path(wav_path)
            x = self.feat(wav).unsqueeze(0).to(self.device)

            logit = self.model(x).squeeze()
            prob_fake = torch.sigmoid(logit).item()
            prob_real = 1.0 - prob_fake
            label = "FAKE" if prob_fake >= 0.5 else "REAL"

            return {
                "label": label,
                "prob_fake": float(prob_fake),
                "prob_real": float(prob_real),
                "converted": converted,
                "original_ext": original_ext,
                "error": None,
            }
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)