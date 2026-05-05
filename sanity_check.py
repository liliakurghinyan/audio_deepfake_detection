from dataset import AudioDeepfakeDataset
from features import MelFeatureExtractor

feat = MelFeatureExtractor(sr=16000, n_mels=80)
ds = AudioDeepfakeDataset("dataset/train", feat)

print("Total audio files:", len(ds))

x, y = ds[0]
print("Feature shape:", x.shape)
print("Label:", y.item(), "(0=real, 1=fake)")
