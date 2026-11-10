"""
Yarışma metriği (oF1): Recod.ai/LUC - Scientific Image Forgery Detection.

Kaynak: Kaggle resmi metrik kodu (https://www.kaggle.com/code/metric/recodai-f1).
Bu dosya, resmi koddaki mantığın RLE kodlaması olmadan, doğrudan binary maskeler
üzerinde çalışan sadeleştirilmiş bir uyarlamasıdır.

Girdi sözleşmesi:
  - Bir görselin tahmini / gerçeği, (H, W) boyutunda binary maskelerin listesidir
    (her eleman bir instance). Authentic görsel için boş liste [] kullanılır.
  - Tahmin maskeleri, gerçek maskelerle AYNI (orijinal) çözünürlükte olmalıdır.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from .data import gt_instances


def f1_binary(pred, gt):
    """İki binary maske arasındaki piksel F1 skoru (2TP / (|pred| + |gt|))."""
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    assert pred.shape == gt.shape, f"Maske boyutları farklı: {pred.shape} vs {gt.shape}"
    tp = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    return float(2 * tp / denom) if denom > 0 else 0.0


def f1_matrix(pred_masks, gt_masks):
    """(tahmin x gerçek) F1 matrisi; tahmin sayısı gerçekten azsa sıfır satırlarla doldurur."""
    m = np.zeros((len(pred_masks), len(gt_masks)))
    for i, p in enumerate(pred_masks):
        for j, g in enumerate(gt_masks):
            m[i, j] = f1_binary(p, g)
    if m.shape[0] < len(gt_masks):
        pad = np.zeros((len(gt_masks) - m.shape[0], len(gt_masks)))
        m = np.vstack([m, pad])
    return m


def oF1_score(pred_masks, gt_masks):
    """Sahte bir görselde instance eşleştirmeli F1 (Macar algoritması)."""
    m = f1_matrix(pred_masks, gt_masks)
    rows, cols = linear_sum_assignment(-m)
    # Fazla tahmin edilen instance'lar için ceza
    penalty = len(gt_masks) / max(len(pred_masks), len(gt_masks))
    return float(m[rows, cols].mean() * penalty)


def image_score(pred_masks, gt_masks):
    """Tek görsel skoru. Boş liste = authentic."""
    if len(gt_masks) == 0 and len(pred_masks) == 0:
        return 1.0      # authentic doğru bilindi
    if len(gt_masks) == 0 or len(pred_masks) == 0:
        return 0.0      # yanlış alarm ya da kaçırma
    return oF1_score(pred_masks, gt_masks)


def evaluate_df(df, predict_fn, return_per_image=False):
    """
    df içindeki her görsel için predict_fn(row) çağırır, skorları hesaplar, ortalamasını döndürür.
    predict_fn(row) -> (H, W) binary maske listesi ([] = authentic), orijinal çözünürlükte.
    Gerçek maskeler görsel görsel okunur, belleği şişirmez.
    """
    scores = []
    for row in tqdm(df.itertuples(index=False), total=len(df)):
        scores.append(image_score(predict_fn(row), gt_instances(row)))
    mean = float(np.mean(scores))
    return (mean, np.array(scores)) if return_per_image else mean