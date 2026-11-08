import pandas as pd
from PIL import Image
from tqdm import tqdm

from .data import load_mask, union_mask


def image_sizes(df):
    """Her görselin genişlik, yükseklik ve etiketini DataFrame olarak döndürür."""
    rows = []
    for path, label in tqdm(zip(df["path"], df["label"]), total=len(df)):
        w, h = Image.open(path).size
        rows.append((w, h, label))
    return pd.DataFrame(rows, columns=["w", "h", "label"])


def mask_stats(df):
    """Sahte görseller için maske alanı oranı ve instance sayısını hesaplar."""
    forg = df[df.label == 1].copy()
    ratios, n_inst = [], []
    for mp in tqdm(forg["mask_path"]):
        m = load_mask(mp)
        n_inst.append(m.shape[0])             # instance sayısı = ilk boyut
        ratios.append(union_mask(m).mean())   # maske alanı / görsel alanı
    forg["area_ratio"] = ratios
    forg["n_inst"] = n_inst
    return forg