import numpy as np
import scipy.linalg

def train_lda(Phi, y, lam=1e-4):
    """
    Train LDA in feature space: theta = Sigma^-1 (mu_+ - mu_-)
    Sigma is the pooled covariance matrix.
    """
    mu_pos = np.mean(Phi[y==1], axis=0)
    mu_neg = np.mean(Phi[y==-1], axis=0)
    
    # Centering
    Phi_c = Phi.copy()
    Phi_c[y==1] -= mu_pos
    Phi_c[y==-1] -= mu_neg
    
    # Pooled Covariance
    Sigma = (Phi_c.T @ Phi_c) / (len(y) - 2)
    
    # Regularization
    Sigma[np.diag_indices_from(Sigma)] += lam

    diff = mu_pos - mu_neg
    
    try:
        theta = scipy.linalg.solve(Sigma, diff, assume_a='pos')
    except:
        # Fallback to least squares if singular
        theta = scipy.linalg.lstsq(Sigma, diff)[0]
        
    return theta
