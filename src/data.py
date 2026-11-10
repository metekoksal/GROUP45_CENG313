import numpy as np
import pandas as pd
from .config import TRAIN_AUTH, TRAIN_FORG, TRAIN_MASK
from sklearn.model_selection import train_test_split

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