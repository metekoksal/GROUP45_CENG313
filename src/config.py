from pathlib import Path

DATA_DIR = Path("/content/data")   # kendi yoluna göre düzelt
TRAIN_AUTH = DATA_DIR / "train_images" / "authentic"
TRAIN_FORG = DATA_DIR / "train_images" / "forged"
TRAIN_MASK = DATA_DIR / "train_masks"