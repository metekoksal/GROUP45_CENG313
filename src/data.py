import numpy as np
import pandas as pd
from .config import TRAIN_AUTH, TRAIN_FORG, TRAIN_MASK

def build_train_df():
    rows = []
    for p in sorted(TRAIN_AUTH.glob("*")):
        rows.append({"case_id": p.stem, "path": str(p), "label": 0, "mask_path": None})
    for p in sorted(TRAIN_FORG.glob("*")):
        rows.append({"case_id": p.stem, "path": str(p), "label": 1,
                     "mask_path": str(TRAIN_MASK / f"{p.stem}.npy")})
    return pd.DataFrame(rows)

def load_mask(path):
    return np.load(path)            # (N, H, W)

def union_mask(mask):
    return mask.max(axis=0)         # (H, W)