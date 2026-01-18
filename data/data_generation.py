import numpy as np

def generate_data(n, d, sigma_scale=1.0, noise=0.0):
    """
    Generate synthetic data with anisotropic covariance.
    
    Args:
        n (int): Number of samples.
        d (int): Input dimension.
        sigma_scale (float): Scale factor for decaying variance.
        noise (float): Label noise rate.
        
    Returns:
        tuple: (X, y) features and labels.
    """
    if sigma_scale == 1.0:
        cov_diag = np.ones(d)
    else:
        cov_diag = np.linspace(1.0, sigma_scale, d)

    X = np.random.randn(n, d) * np.sqrt(cov_diag)

    beta = np.zeros(d)
    beta[0] = 1.0

    y = np.sign(X @ beta)

    if noise > 0:
        flip = np.random.rand(n) < noise
        y[flip] *= -1

    return X, y
