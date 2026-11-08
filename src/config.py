from pathlib import Path

DATA_DIR = Path("/content/data")

TRAIN_IMG_DIR   = DATA_DIR / "train_images"
TRAIN_AUTH      = TRAIN_IMG_DIR / "authentic"
TRAIN_FORG      = TRAIN_IMG_DIR / "forged"
TRAIN_MASK      = DATA_DIR / "train_masks"

SUPP_IMG        = DATA_DIR / "supplemental_images"
SUPP_MASK       = DATA_DIR / "supplemental_masks"

TEST_IMG        = DATA_DIR / "test_images"
SAMPLE_SUB      = DATA_DIR / "sample_submission.csv"