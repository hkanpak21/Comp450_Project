import numpy as np

def train_generative(Phi, y):
    """
    Train a Nearest Centroid Model (Generative Model).
    
    Args:
        Phi (np.ndarray): Training features (n, N).
        y (np.ndarray): Training labels (n,).
        
    Returns:
        np.ndarray: Model weights (N,).
    """
    pos_mask = (y == 1)
    neg_mask = (y == -1)

    if np.sum(pos_mask) == 0: return -np.ones(Phi.shape[1])
    if np.sum(neg_mask) == 0: return np.ones(Phi.shape[1])

    mu_pos = np.mean(Phi[pos_mask], axis=0)
    mu_neg = np.mean(Phi[neg_mask], axis=0)
    theta = mu_pos - mu_neg
    return theta
