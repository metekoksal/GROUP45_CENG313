import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .data import load_mask, union_mask

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