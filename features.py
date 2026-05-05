import torchaudio

class MelFeatureExtractor:
    def __init__(self, sr=16000, n_mels=80):
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=1024,
            hop_length=256,
            n_mels=n_mels
        )
        self.db = torchaudio.transforms.AmplitudeToDB()

    def __call__(self, wav):
        mel = self.mel(wav)        # (1, n_mels, time)
        mel_db = self.db(mel)
        return mel_db
