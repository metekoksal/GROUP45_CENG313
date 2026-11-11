import cv2
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from .data import ForgeryDataset, gt_instances
from .utils import image_score

"""Baseline 0: Trivial Model"""
def predict_all_authentic(df):
    """Her görsele 'authentic' (0) der."""
    return np.zeros(len(df), dtype=int)


"""Baseline 1: SIFT + RANSAC ile klasik copy-move tespiti (eğitim gerektirmez)."""
class SiftCopyMoveDetector:
    """
    Kullanım: det = SiftCopyMoveDetector(); masks = det(row)
    Dönüş: orijinal çözünürlükte (H, W) uint8 maske listesi ([] = authentic).
    """

    def __init__(self, max_side=1024, n_features=3000, ratio=0.6, min_dist=20,
                 min_inliers=10, ransac_thr=3.0, max_instances=3,
                 radius=10, split_regions=False):
        self.max_side = max_side
        self.ratio = ratio                  # Lowe ratio testi eşiği (küçük = daha katı)
        self.min_dist = min_dist            # birbirine bu kadar yakın noktalar eşleşmez (px)
        self.min_inliers = min_inliers      # "sahte" demek için gereken en az tutarlı eşleşme
        self.ransac_thr = ransac_thr
        self.max_instances = max_instances
        self.radius = radius                # maske üretiminde keypoint çevresi (px)
        self.split_regions = split_regions  # True: kaynak ve kopya bölge ayrı instance
        self.sift = cv2.SIFT_create(nfeatures=n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

    def __call__(self, row):
        img = np.array(Image.open(row.path).convert("RGB"))
        return self.predict(img)

    def predict(self, img_rgb):
        h, w = img_rgb.shape[:2]
        scale = min(1.0, self.max_side / max(h, w))
        small = cv2.resize(img_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) \
            if scale < 1 else img_rgb
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)

        kps, des = self.sift.detectAndCompute(gray, None)
        if des is None or len(kps) < 2 * self.min_inliers:
            return []

        pts = np.array([k.pt for k in kps], dtype=np.float32)
        pairs = self._match_pairs(pts, des)
        small_masks = self._cluster_masks(pts, pairs, gray.shape)

        return [cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) for m in small_masks]

    def _match_pairs(self, pts, des):
        """Her keypoint için, yeterince uzaktaki en iyi komşuyla ratio testini geçen eşleşmeler."""
        knn = self.matcher.knnMatch(des, des, k=6)
        pairs = []
        for i, ms in enumerate(knn):
            far = [m for m in ms if np.linalg.norm(pts[m.trainIdx] - pts[i]) >= self.min_dist]
            if len(far) >= 2 and far[0].distance < self.ratio * far[1].distance:
                pairs.append((i, far[0].trainIdx))
        return pairs

    def _cluster_masks(self, pts, pairs, shape):
        """RANSAC'i tekrarlayarak her geometrik dönüşüm grubundan bir maske üretir."""
        masks, remaining = [], list(pairs)
        for _ in range(self.max_instances):
            if len(remaining) < self.min_inliers:
                break
            src = pts[[i for i, _ in remaining]]
            dst = pts[[j for _, j in remaining]]
            M, inl = cv2.estimateAffinePartial2D(
                src, dst, method=cv2.RANSAC, ransacReprojThreshold=self.ransac_thr)
            if M is None or inl is None:
                break
            inl = inl.ravel().astype(bool)
            if inl.sum() < self.min_inliers:
                break

            used = [p for p, ok in zip(remaining, inl) if ok]
            idxs = {k for p in used for k in p}
            masks.extend(self._to_masks(pts, idxs, shape))

            # kullanılan eşleşmeleri ve ters yönlerini (j, i) çıkar; aksi halde aynı bölge 2. kez bulunur
            drop = set(used) | {(j, i) for i, j in used}
            remaining = [p for p in remaining if p not in drop]
        return masks

    def _to_masks(self, pts, idxs, shape):
        m = np.zeros(shape, np.uint8)
        for k in idxs:
            cv2.circle(m, (int(pts[k][0]), int(pts[k][1])), self.radius, 1, -1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * self.radius + 1,) * 2)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        if not self.split_regions:
            return [m]
        n, lab = cv2.connectedComponents(m)
        return [(lab == c).astype(np.uint8) for c in range(1, n)]

"""Baseline 2: U-net"""
def build_unet(encoder="resnet34", pretrained=True):
    """ImageNet ön-eğitimli encoder'lı U-Net, çıktı: (B, 1, H, W) logit."""
    return smp.Unet(encoder_name=encoder,
                    encoder_weights="imagenet" if pretrained else None,
                    in_channels=3, classes=1)


def _run_epoch(model, loader, loss_fn, device, opt=None, scaler=None):
    train = opt is not None
    model.train(train)
    total, n = 0.0, 0
    for b in tqdm(loader, leave=False):
        x, y = b["image"].to(device), b["mask"].to(device)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                loss = loss_fn(model(x), y)
        if train:
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        total += loss.item() * len(x)
        n += len(x)
    return total / n


"""BCE + Dice kayıp fonksiyonu (az pozitif piksel için)."""
class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, eps=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.eps = eps

    def forward(self, logits, target):
        logits, target = logits.float(), target.float()
        bce = F.binary_cross_entropy_with_logits(logits, target)
        p = torch.sigmoid(logits)
        inter = (p * target).sum(dim=(1, 2, 3))
        denom = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = 1 - (2 * inter + self.eps) / (denom + self.eps)   # boş maskede yanlış pozitifi cezalandırır
        return self.bce_weight * bce + (1 - self.bce_weight) * dice.mean()


def fit(model, train_df, val_df, epochs=10, batch_size=8, lr=3e-4, size=512,
        save_path="/content/checkpoints/unet.pth", num_workers=2):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    tr = DataLoader(ForgeryDataset(train_df, size, augment=True), batch_size=batch_size,
                    shuffle=True, num_workers=num_workers, pin_memory=True)
    va = DataLoader(ForgeryDataset(val_df, size), batch_size=batch_size,
                    shuffle=False, num_workers=num_workers, pin_memory=True)
    loss_fn = BCEDiceLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler(device, enabled=device == "cuda")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    best, history = float("inf"), []
    for ep in range(1, epochs + 1):
        tl = _run_epoch(model, tr, loss_fn, device, opt, scaler)
        vl = _run_epoch(model, va, loss_fn, device)
        history.append({"epoch": ep, "train_loss": tl, "val_loss": vl})
        print(f"epoch {ep}/{epochs} | train {tl:.4f} | val {vl:.4f}")
        if vl < best:
            best = vl
            torch.save(model.state_dict(), save_path)
    return history

# U-net post processing
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