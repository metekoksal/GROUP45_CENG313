import numpy as np

def predict_all_authentic(df):
    """Her görsele 'authentic' (0) der."""
    return np.zeros(len(df), dtype=int)