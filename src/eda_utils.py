import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from .data import load_mask, union_mask
import numpy as np
from matplotlib.colors import ListedColormap

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


_COLORS = ["red", "cyan", "yellow", "lime", "magenta"]


def _show_rows(rows, cols=3):
    """Verilen satırları (görsel + varsa maske overlay'i) ızgarada çizer."""
    rows = rows.reset_index(drop=True)
    n_rows = int(np.ceil(len(rows) / cols))
    fig, axes = plt.subplots(n_rows, cols, figsize=(5 * cols, 4 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (_, r) in zip(axes, rows.iterrows()):
        img = np.array(Image.open(r["path"]).convert("RGB"))
        ax.imshow(img)
        if r["label"] == 1:
            mask = load_mask(r["mask_path"])
            h, w = img.shape[:2]
            for i, inst in enumerate(mask):
                ax.imshow(np.ma.masked_where(inst == 0, inst),
                          cmap=ListedColormap([_COLORS[i % len(_COLORS)]]),
                          alpha=0.5, vmin=0, vmax=1,
                          extent=(0, w, h, 0))   # maske boyutu farklı olsa da görsele oturur
            area = union_mask(mask).mean() * 100
            ax.set_title(f"{r['case_id']} | {mask.shape[0]} instance | %{area:.1f} alan")
        else:
            ax.set_title(f"{r['case_id']} | authentic")
        ax.axis("off")

    for ax in axes[len(rows):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def show_samples(df, n=6, label=1, seed=42, cols=3):
    """Rastgele n örneği çizer. label=1 sahte (maskeli), label=0 authentic."""
    sub = df[df.label == label]
    sample = sub.sample(n=min(n, len(sub)), random_state=seed)
    return _show_rows(sample, cols)


def show_extremes(forg, k=3):
    """En küçük ve en büyük maskeli k sahte görseli çizer (forg = mask_stats çıktısı)."""
    small = forg.nsmallest(k, "area_ratio")
    large = forg.nlargest(k, "area_ratio")
    return _show_rows(pd.concat([small, large]), cols=k)


def plot_size_by_class(size_df, axes=None):
    """Authentic ve forged görsellerin genişlik/yükseklik dağılımını kutu grafiğiyle karşılaştırır."""
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, title in zip(axes, ["w", "h"], ["Genişlik (px)", "Yükseklik (px)"]):
        data = [size_df.loc[size_df.label == k, col] for k in (0, 1)]
        ax.boxplot(data)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["authentic", "forged"])
        ax.set_yscale("log")
        ax.set_title(title)
    return axes


def top_sizes_by_class(size_df, k=8):
    """En sık k çözünürlük için authentic/forged sayıları ve sahte oranı."""
    t = (size_df.assign(size=size_df["w"].astype(str) + "x" + size_df["h"].astype(str))
                .groupby(["size", "label"]).size().unstack(fill_value=0)
                .rename(columns={0: "authentic", 1: "forged"}))
    t["total"] = t["authentic"] + t["forged"]
    t["forged_ratio"] = (t["forged"] / t["total"]).round(2)
    return t.sort_values("total", ascending=False).head(k)