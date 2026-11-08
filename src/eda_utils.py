import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
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

def plot_area_ratio(forg, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.hist(forg["area_ratio"] * 100, bins=50)
    ax.axvline(forg["area_ratio"].median() * 100, color="red", ls="--", label="medyan")
    ax.set_xscale("log")
    ax.set_xlabel("Maske alanı (% görsel alanı, log ölçek)")
    ax.set_ylabel("Görsel sayısı")
    ax.set_title("Maske alanı dağılımı")
    ax.legend()
    return ax

def plot_size_scatter(size_df, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    for label, name in [(0, "authentic"), (1, "forged")]:
        s = size_df[size_df.label == label]
        ax.scatter(s["w"], s["h"], s=8, alpha=0.4, label=name)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Genişlik (px)"); ax.set_ylabel("Yükseklik (px)")
    ax.set_title("Görsel boyutları")
    ax.legend()
    return ax

def plot_instance_counts(forg, ax=None):
    """Sahte görsellerde instance sayısı dağılımını, sayı ve yüzdelerle çizer."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))

    counts = forg["n_inst"].value_counts().sort_index()
    total = counts.sum()

    bars = ax.bar(counts.index.astype(str), counts.values)

    # Çubukların üstüne "sayı (yüzde)" yaz
    labels = [f"{c}\n(%{c / total * 100:.1f})" for c in counts.values]
    ax.bar_label(bars, labels=labels, padding=3)

    ax.set_ylim(0, counts.max() * 1.15)   # etiketlere yer aç
    ax.set_xlabel("Instance sayısı")
    ax.set_ylabel("Görsel sayısı")
    ax.set_title(f"Sahte görsellerde instance sayısı dağılımı (n={total})")
    return ax