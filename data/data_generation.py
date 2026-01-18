import numpy as np

def generate_data(n, d, sigma_scale=1.0, noise=0.0, seed=None):
    """
    Generate synthetic binary classification data.
    
    Args:
        n: Number of samples
        d: Input dimension
        sigma_scale: Scaling factor for data covariance (anisotropy)
        noise: Probability of flipping labels
        seed: Random seed for reproducibility
        
    Returns:
        X: (n, d) feature matrix
        y: (n,) label vector (values in {-1, 1})
    """
    if seed is not None:
        np.random.seed(seed)

    if sigma_scale == 1.0:
        cov_diag = np.ones(d)
    else:
        # Anisotropic: High variance in noise directions
        cov_diag = np.linspace(1.0, sigma_scale, d)

    X = np.random.randn(n, d) * np.sqrt(cov_diag)
    
    # True signal: Beta aligned with first dimension
    beta = np.zeros(d)
    beta[0] = 1.0
    
    y = np.sign(X @ beta)

    # Add label noise
    if noise > 0:
        flip = np.random.rand(n) < noise
        y[flip] *= -1
        
    return X, y
