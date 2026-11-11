"""Baseline 1: SIFT + RANSAC ile klasik copy-move tespiti (eğitim gerektirmez)."""
import cv2
import numpy as np
from PIL import Image


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