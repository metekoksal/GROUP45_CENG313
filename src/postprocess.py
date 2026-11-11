"""U-Net olasılık haritalarını instance maskelerine çevirir ve yarışma metriğiyle puanlar."""
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import gt_instances
from .dataset import ForgeryDataset
from .metrics import image_score


@torch.no_grad()
def predict_probs(model, df, size=512, batch_size=16, num_workers=2):
    """
    df'deki her görsel için (sırayla) olasılık haritası üretir.
    Döndürür: probs (N, size, size) uint8 [0-255], sizes (N, 2) orijinal (h, w).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    dl = DataLoader(ForgeryDataset(df, size, augment=False), batch_size=batch_size,
                    shuffle=False, num_workers=num_workers)
    probs, sizes = [], []
    for b in tqdm(dl):
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
            logits = model(b["image"].to(device))
        p = torch.sigmoid(logits.float())[:, 0]
        probs.append((p * 255).round().byte().cpu().numpy())
        sizes.append(np.stack([b["h"].numpy(), b["w"].numpy()], axis=1))
    return np.concatenate(probs), np.concatenate(sizes)


def probs_to_instances(prob_u8, h, w, thr=0.5, min_area_frac=0.002,
                       min_score=0.5, max_instances=5):
    """
    Tek bir olasılık haritasından orijinal çözünürlükte (h, w) instance maskeleri listesi üretir.
    Boş liste = 'authentic'.
    """
    p = prob_u8.astype(np.float32) / 255.0
    binary = (p >= thr).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    min_area = min_area_frac * binary.size
    comps = []
    for c in range(1, n):
        area = stats[c, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        m = lab == c
        if p[m].mean() < min_score:
            continue
        comps.append((area, m))

    comps.sort(key=lambda t: -t[0])                  # büyükten küçüğe
    return [cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            for _, m in comps[:max_instances]]


def score_probs(df, probs, sizes, idx=None, gt_cache=None, return_per_image=False, **params):
    """
    Olasılık haritalarını verilen ayarlarla instance'lara çevirip oF1 skorunu hesaplar.
    idx: df içindeki konumlar (None = hepsi). gt_cache: {konum: gerçek maskeler} sözlüğü,
    ayar aramasında maskeleri tekrar tekrar diskten okumamak için.
    """
    idx = np.arange(len(df)) if idx is None else np.asarray(idx)
    rows = list(df.itertuples(index=False))
    scores = []
    for i in idx:
        if gt_cache is not None:
            if i not in gt_cache:
                gt_cache[i] = gt_instances(rows[i])
            gt = gt_cache[i]
        else:
            gt = gt_instances(rows[i])
        h, w = sizes[i]
        pred = probs_to_instances(probs[i], int(h), int(w), **params)
        scores.append(image_score(pred, gt))
    mean = float(np.mean(scores))
    return (mean, np.array(scores)) if return_per_image else mean