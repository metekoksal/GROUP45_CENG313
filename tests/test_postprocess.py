import numpy as np
from src.models import probs_to_instances


def blob(prob, r0, r1, c0, c1, v=230):
    prob[r0:r1, c0:c1] = v
    return prob


def test_empty_map_is_authentic():
    assert probs_to_instances(np.zeros((512, 512), np.uint8), 512, 512) == []

def test_single_blob():
    p = blob(np.zeros((512, 512), np.uint8), 100, 200, 100, 200)
    out = probs_to_instances(p, 512, 512)
    assert len(out) == 1 and out[0].shape == (512, 512)

def test_two_blobs_two_instances():
    p = np.zeros((512, 512), np.uint8)
    blob(p, 50, 150, 50, 150); blob(p, 300, 400, 300, 400)
    assert len(probs_to_instances(p, 512, 512)) == 2

def test_tiny_blob_removed():
    p = blob(np.zeros((512, 512), np.uint8), 10, 14, 10, 14)    # 16 piksel
    assert probs_to_instances(p, 512, 512, min_area_frac=0.002) == []

def test_low_confidence_removed():
    p = blob(np.zeros((512, 512), np.uint8), 100, 200, 100, 200, v=140)   # ~0.55
    assert probs_to_instances(p, 512, 512, thr=0.5, min_score=0.7) == []

def test_resized_to_original():
    p = blob(np.zeros((512, 512), np.uint8), 100, 200, 100, 200)
    out = probs_to_instances(p, 1000, 666)
    assert out[0].shape == (1000, 666)