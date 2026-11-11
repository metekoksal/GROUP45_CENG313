"""BCE + Dice kayıp fonksiyonu (az pozitif piksel için)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


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