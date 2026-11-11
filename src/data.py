import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset



"""Config"""
DATA_DIR = Path("/content/data")

TRAIN_IMG_DIR   = DATA_DIR / "train_images"
TRAIN_AUTH      = TRAIN_IMG_DIR / "authentic"
TRAIN_FORG      = TRAIN_IMG_DIR / "forged"
TRAIN_MASK      = DATA_DIR / "train_masks"

SUPP_IMG        = DATA_DIR / "supplemental_images"
SUPP_MASK       = DATA_DIR / "supplemental_masks"

TEST_IMG        = DATA_DIR / "test_images"
SAMPLE_SUB      = DATA_DIR / "sample_submission.csv"



def build_train_df():
    rows = []
    for p in sorted(TRAIN_AUTH.glob("*")):
        rows.append({"case_id": p.stem, "path": str(p), "label": 0, "mask_path": None})
    for p in sorted(TRAIN_FORG.glob("*")):
        rows.append({"case_id": p.stem, "path": str(p), "label": 1,
                     "mask_path": str(TRAIN_MASK / f"{p.stem}.npy")})
    return pd.DataFrame(rows)

def load_mask(path):
    return np.load(path)        # (N, H, W)

def union_mask(mask):
    return mask.max(axis=0)     # (H, W)

def split_train_val(df, val_size=0.2, seed=42):
    """Authentic/forged oranını koruyarak train ve validation'a böler."""
    train_df, val_df = train_test_split(
        df, test_size=val_size, stratify=df["label"], random_state=seed
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

def gt_instances(row):
    """Gerçek instance maskelerini liste olarak döndürür; authentic için boş liste."""
    if row.label == 0:
        return []
    return list(load_mask(row.mask_path))


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

class ForgeryDataset(Dataset):
    """
    df: build_train_df / split_train_val çıktısı (path, label, mask_path, case_id sütunları).
    Döndürür: {"image": (3,S,S) float, "mask": (1,S,S) float 0/1, "label": 0/1,
               "h": orijinal yükseklik, "w": orijinal genişlik, "case_id": str}
    """

    def __init__(self, df, size=512, augment=False):
        self.df = df.reset_index(drop=True)
        self.size = size
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = np.array(Image.open(row["path"]).convert("RGB"))
        h, w = img.shape[:2]

        if row["label"] == 1:
            mask = union_mask(load_mask(row["mask_path"])).astype(np.uint8)
        else:
            mask = np.zeros((h, w), dtype=np.uint8)

        interp = cv2.INTER_AREA if max(h, w) > self.size else cv2.INTER_LINEAR
        img = cv2.resize(img, (self.size, self.size), interpolation=interp)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            if torch.rand(1).item() < 0.5:                  # yatay çevirme
                img, mask = img[:, ::-1], mask[:, ::-1]
            if torch.rand(1).item() < 0.5:                  # dikey çevirme
                img, mask = img[::-1], mask[::-1]
            img, mask = np.ascontiguousarray(img), np.ascontiguousarray(mask)

        x = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(x.transpose(2, 0, 1).copy())
        y = torch.from_numpy(mask.astype(np.float32))[None]

        return {"image": x, "mask": y, "label": int(row["label"]),
                "h": h, "w": w, "case_id": str(row["case_id"])}