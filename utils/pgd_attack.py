import numpy as np

def pgd_attack(model_theta, W, X, y, epsilon, alpha, steps):
    """
    Projected Gradient Descent (PGD) Attack.
    
    Args:
        model_theta (np.ndarray): Model weights.
        W (np.ndarray): Random weights for feature projection.
        X (np.ndarray): Original features.
        y (np.ndarray): Labels.
        epsilon (float): Maximum perturbation magnitude (L2 norm).
        alpha (float): Step size.
        steps (int): Number of steps.
        
    Returns:
        np.ndarray: Adversarial features.
    """
    N = W.shape[0]
    X_adv = X.copy()

    for _ in range(steps):
        Z_pre = X_adv @ W.T
        activations = (Z_pre > 0).astype(float)

        # Gradient w.r.t input
        grad_x = (activations * model_theta[None, :]) @ W
        grad_x = grad_x / np.sqrt(N)

        step_dir = -y[:, None] * np.sign(grad_x)
        X_adv = X_adv + alpha * step_dir

        # Projection
        delta = X_adv - X
        norms = np.linalg.norm(delta, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        factor = np.minimum(1, epsilon / norms)
        delta = delta * factor
        X_adv = X + delta

    return X_adv
