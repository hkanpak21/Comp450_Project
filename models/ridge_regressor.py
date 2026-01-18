import numpy as np
import scipy.linalg

def train_ridge(Phi, y, lam=1e-8):
    """
    Train a Ridge Regressor (Discriminative Model).
    
    Args:
        Phi (np.ndarray): Training features (n, N).
        y (np.ndarray): Training labels (n,).
        lam (float): Regularization parameter.
        
    Returns:
        np.ndarray: Trained model weights (N,).
    """
    n, N = Phi.shape
    if N > n:
        K = Phi @ Phi.T
        K[np.diag_indices_from(K)] += lam
        try:
            alpha = scipy.linalg.solve(K, y, assume_a='pos')
        except scipy.linalg.LinAlgError:
            alpha = scipy.linalg.lstsq(K, y)[0]
        theta = Phi.T @ alpha
    else:
        Cov = Phi.T @ Phi
        Cov[np.diag_indices_from(Cov)] += lam
        try:
            theta = scipy.linalg.solve(Cov, Phi.T @ y, assume_a='pos')
        except:
            theta = scipy.linalg.lstsq(Cov, Phi.T @ y)[0]
    return theta
