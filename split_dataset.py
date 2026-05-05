import random
import shutil
from pathlib import Path

REAL_SRC = Path("real new")
FAKE_SRC = Path("fake new")
OUT_DIR = Path("dataset")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

AUDIO_EXTS = [".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"]

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


def collect_files(folder):
    files = []
    for ext in AUDIO_EXTS:
        files.extend(folder.glob(f"*{ext}"))
        files.extend(folder.glob(f"*{ext.upper()}"))
    return files


def clear_old_dataset():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)


def make_dirs():
    for split in ["train", "val", "test"]:
        for label in ["real", "fake"]:
            (OUT_DIR / split / label).mkdir(parents=True, exist_ok=True)


def random_split(files):
    files = list(files)
    random.shuffle(files)

    n = len(files)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    return train_files, val_files, test_files


def copy_files(files, split, label):
    for file in files:
        dst = OUT_DIR / split / label / file.name
        shutil.copy2(file, dst)


def process_class(src_folder, label):
    files = collect_files(src_folder)

    if not files:
        raise RuntimeError(f"No audio files found in folder: {src_folder}")

    train_files, val_files, test_files = random_split(files)

    copy_files(train_files, "train", label)
    copy_files(val_files, "val", label)
    copy_files(test_files, "test", label)

    print(f"\n{label.upper()} files")
    print(f"Total : {len(files)}")
    print(f"Train : {len(train_files)}")
    print(f"Val   : {len(val_files)}")
    print(f"Test  : {len(test_files)}")


def main():
    clear_old_dataset()
    make_dirs()

    process_class(REAL_SRC, "real")
    process_class(FAKE_SRC, "fake")

    print("\nDone.")
    print("Random dataset split created in:")
    print("dataset/train/real")
    print("dataset/train/fake")
    print("dataset/val/real")
    print("dataset/val/fake")
    print("dataset/test/real")
    print("dataset/test/fake")


if __name__ == "__main__":
    main()