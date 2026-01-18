import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from data.data_generation import generate_data
from models.random_features import get_random_features
from models.ridge_regressor import train_ridge
from models.generative_model import train_generative
from utils.pgd_attack import pgd_attack
import config

def run_experiment_A():
    """Isotropic Data, Varying N (The Curse)"""
    print("\n>>> STARTING EXPERIMENT A: The Curse (Isotropic Data) <<<")

    X_train, y_train = generate_data(config.N_TRAIN, config.D, sigma_scale=1.0, noise=config.NOISE_RATE)
    X_test, y_test = generate_data(config.N_TEST, config.D, sigma_scale=1.0, noise=0.0)

    N_values = np.linspace(config.D, 2000, 10).astype(int)
    data = []

    for N in N_values:
        W = np.random.randn(N, config.D)
        Phi_train = get_random_features(X_train, W)
        Phi_test = get_random_features(X_test, W)

        theta_A = train_ridge(Phi_train, y_train, lam=1e-6)
        theta_B = train_generative(Phi_train, y_train)

        acc_A = accuracy_score(y_test, np.sign(Phi_test @ theta_A))
        acc_B = accuracy_score(y_test, np.sign(Phi_test @ theta_B))

        X_adv_A = pgd_attack(theta_A, W, X_test, y_test, config.EPSILON, config.PGD_ALPHA, config.PGD_STEPS)
        rob_A = accuracy_score(y_test, np.sign(get_random_features(X_adv_A, W) @ theta_A))

        X_adv_B = pgd_attack(theta_B, W, X_test, y_test, config.EPSILON, config.PGD_ALPHA, config.PGD_STEPS)
        rob_B = accuracy_score(y_test, np.sign(get_random_features(X_adv_B, W) @ theta_B))

        norm_A = np.linalg.norm(theta_A)
        norm_B = np.linalg.norm(theta_B)

        data.append({
            'N': N, 'Ratio': N/config.N_TRAIN,
            'Acc_A': acc_A, 'Rob_A': rob_A, 'Norm_A': norm_A,
            'Acc_B': acc_B, 'Rob_B': rob_B, 'Norm_B': norm_B
        })
        print(f"N={N}: Rob_A={rob_A:.2f}, Rob_B={rob_B:.2f}, Norm_A={norm_A:.1f}, Norm_B={norm_B:.1f}")

    return pd.DataFrame(data)

def run_experiment_B():
    """Regularization Sweep (The Tradeoff)"""
    print("\n>>> STARTING EXPERIMENT B: Regularization Pareto <<<")

    N_FIXED = 2000
    X_train, y_train = generate_data(config.N_TRAIN, config.D, sigma_scale=1.0, noise=config.NOISE_RATE)
    X_test, y_test = generate_data(config.N_TEST, config.D, noise=0.0)

    W = np.random.randn(N_FIXED, config.D)
    Phi_train = get_random_features(X_train, W)
    Phi_test = get_random_features(X_test, W)

    theta_B = train_generative(Phi_train, y_train)
    acc_B = accuracy_score(y_test, np.sign(Phi_test @ theta_B))

    X_adv_B = pgd_attack(theta_B, W, X_test, y_test, config.EPSILON, config.PGD_ALPHA, config.PGD_STEPS)
    rob_B = accuracy_score(y_test, np.sign(get_random_features(X_adv_B, W) @ theta_B))

    print(f"Generative Point: Acc={acc_B:.3f}, Rob={rob_B:.3f}")

    lambdas = np.logspace(-6, 2, 10)
    data = []

    for lam in lambdas:
        theta_A = train_ridge(Phi_train, y_train, lam=lam)
        acc_A = accuracy_score(y_test, np.sign(Phi_test @ theta_A))

        X_adv = pgd_attack(theta_A, W, X_test, y_test, config.EPSILON, config.PGD_ALPHA, config.PGD_STEPS)
        rob_A = accuracy_score(y_test, np.sign(get_random_features(X_adv, W) @ theta_A))

        data.append({
            'Lambda': lam,
            'Acc_A': acc_A,
            'Rob_A': rob_A
        })
        print(f"Lam={lam:.1e}: Acc={acc_A:.3f}, Rob={rob_A:.3f}")

    return pd.DataFrame(data), (acc_B, rob_B)

def run_experiment_C():
    """Anisotropic Data (The Hidden Cost)"""
    print("\n>>> STARTING EXPERIMENT C: Anisotropic Cost <<<")

    SIGMA = 25.0

    X_train, y_train = generate_data(config.N_TRAIN, config.D, sigma_scale=SIGMA, noise=config.NOISE_RATE)
    X_test, y_test = generate_data(config.N_TEST, config.D, sigma_scale=SIGMA, noise=0.0)

    N_values = np.linspace(config.D, 2000, 10).astype(int)
    data = []

    for N in N_values:
        W = np.random.randn(N, config.D)
        Phi_train = get_random_features(X_train, W)
        Phi_test = get_random_features(X_test, W)

        theta_A = train_ridge(Phi_train, y_train, lam=1e-6)
        theta_B = train_generative(Phi_train, y_train)

        acc_A = accuracy_score(y_test, np.sign(Phi_test @ theta_A))
        acc_B = accuracy_score(y_test, np.sign(Phi_test @ theta_B))

        data.append({
            'N': N,
            'Acc_A': acc_A,
            'Acc_B': acc_B
        })
        print(f"N={N}: Acc_A={acc_A:.2f}, Acc_B={acc_B:.2f}")

    return pd.DataFrame(data)
