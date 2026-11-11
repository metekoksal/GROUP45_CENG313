import numpy as np
from src.utils import image_score


def box(r0, r1, c0, c1, h=20, w=20):
    m = np.zeros((h, w), dtype=np.uint8)
    m[r0:r1, c0:c1] = 1
    return m


def test_both_authentic():
    assert image_score([], []) == 1.0

def test_false_alarm():
    assert image_score([box(0, 5, 0, 5)], []) == 0.0

def test_missed_forgery():
    assert image_score([], [box(0, 5, 0, 5)]) == 0.0

def test_perfect_match():
    assert image_score([box(0, 5, 0, 5)], [box(0, 5, 0, 5)]) == 1.0

def test_missing_instance():
    gt = [box(0, 5, 0, 5), box(10, 15, 10, 15)]
    assert image_score([box(0, 5, 0, 5)], gt) == 0.5

def test_extra_instance():
    gt = [box(0, 5, 0, 5)]
    pred = [box(0, 5, 0, 5), box(10, 15, 10, 15)]
    assert image_score(pred, gt) == 0.5     # eşleşme 1.0 x ceza 1/2

def test_order_does_not_matter():
    gt = [box(0, 5, 0, 5), box(10, 15, 10, 15)]
    assert image_score(gt[::-1], gt) == 1.0