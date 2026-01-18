import numpy as np

def get_random_features(X, W):
    """
    Project input features into a random feature space using ReLU.
    
    Args:
        X (np.ndarray): Input features (n, d).
        W (np.ndarray): Random weights (N, d).
        
    Returns:
        np.ndarray: Projected features (n, N) scaled by 1/sqrt(N).
    """
    N = W.shape[0]
    Z = np.maximum(0, X @ W.T)
    return Z / np.sqrt(N)
