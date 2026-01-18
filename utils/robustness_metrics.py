import numpy as np
from sklearn.metrics import accuracy_score
from models.random_features import get_random_features

def auto_attack(model_theta, W, X, y, epsilon, steps=10, restarts=3):
    """
    Simplified AutoAttack: PGD on Margin (Minimize y*f(x)) with random restarts.
    """
    N = W.shape[0]
    X_adv_best = X.copy()
    # We want to minimize margin y * f(x), or maximize loss -y * f(x)
    loss_best = -1e9 * np.ones(len(y))

    for r in range(restarts):
        # Random initial perturbation
        X_adv = X + np.random.uniform(-epsilon, epsilon, X.shape)
        
        for _ in range(steps):
            Z = np.maximum(0, X_adv @ W.T)
            f_x = (Z @ model_theta) / np.sqrt(N)
            margin = y * f_x
            
            # Update best adversarial examples found so far
            neg_margin = -margin
            improved = neg_margin > loss_best
            X_adv_best[improved] = X_adv[improved]
            loss_best[improved] = neg_margin[improved]

            # Gradient of f(x)
            activations = (Z > 0).astype(float)
            grad_x = (activations * model_theta[None, :]) @ W / np.sqrt(N)

            # Step: Move against gradient of margin (y * f(x))
            # Descent direction for margin: -y * grad_f
            step = -y[:, None] * grad_x
            X_adv = X_adv + 0.05 * np.sign(step)

            # Projection to L_inf or L2 (User snippet used L2-like clipping/projection)
            delta = X_adv - X
            norms = np.linalg.norm(delta, axis=1, keepdims=True) + 1e-10
            X_adv = X + delta * np.minimum(1, epsilon/norms)

    return X_adv_best

def certify(model_theta, W, X, y, sigma_smooth=0.2, n_samples=100):
    """
    Randomized smoothing certification (Cohen et al. 2019).
    Returns the certified accuracy at radius R (proxy: majority vote confidence).
    """
    N = W.shape[0]
    certified = 0

    for i in range(len(X)):
        # Monte Carlo sampling: f_smooth(x) = E[ sign( f(x + noise) ) ]
        noise = np.random.randn(n_samples, X.shape[1]) * sigma_smooth
        X_noisy = X[i] + noise
        Phi_noisy = np.maximum(0, X_noisy @ W.T) / np.sqrt(N)
        preds = np.sign(Phi_noisy @ model_theta)
        
        # Majority vote
        counts = np.sum(preds == y[i])
        p_A = counts / n_samples
        
        # If confidence is high, consider it certified correctly
        # p_A > 0.9 is a proxy for radius > 0 given sigma
        if p_A > 0.9:
            certified += 1

    return certified / len(X)
