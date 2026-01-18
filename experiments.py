import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from data.data_generation import generate_data
from models.random_features import get_random_features
from models.ridge_regressor import train_ridge
from models.generative_model import train_generative
from utils.pgd_attack import pgd_attack
from models.lda_model import train_lda
from data.real_data_loaders import load_mnist_binary, get_cifar_features
from utils.robustness_metrics import auto_attack, certify
import config

def run_single_trial(N_features, lam_ridge, sigma_scale, noise_rate, seed, use_auto_attack=False, include_lda=False):
    """Run a single experiment trial with specific parameters."""
    # 1. Data
    X_train, y_train = generate_data(config.N_TRAIN, config.D, sigma_scale, noise=noise_rate, seed=seed)
    X_test, y_test   = generate_data(config.N_TEST, config.D, sigma_scale, noise=0.0, seed=None)

    # 2. Features
    W = np.random.randn(N_features, config.D)
    Phi_train = get_random_features(X_train, W)
    Phi_test  = get_random_features(X_test, W)

    # 3. Models
    theta_A = train_ridge(Phi_train, y_train, lam=lam_ridge)
    theta_B = train_generative(Phi_train, y_train)
    theta_L = None
    if include_lda:
        theta_L = train_lda(Phi_train, y_train)

    # 4. Metrics
    norm_A = np.linalg.norm(theta_A)
    norm_B = np.linalg.norm(theta_B)

    acc_A = accuracy_score(y_test, np.sign(Phi_test @ theta_A))
    acc_B = accuracy_score(y_test, np.sign(Phi_test @ theta_B))

    if use_auto_attack:
        X_adv_A = auto_attack(theta_A, W, X_test, y_test, config.EPSILON, config.AUTO_ATTACK_STEPS, config.AUTO_ATTACK_RESTARTS)
        rob_A   = accuracy_score(y_test, np.sign(get_random_features(X_adv_A, W) @ theta_A))
        X_adv_B = auto_attack(theta_B, W, X_test, y_test, config.EPSILON, config.AUTO_ATTACK_STEPS, config.AUTO_ATTACK_RESTARTS)
        rob_B   = accuracy_score(y_test, np.sign(get_random_features(X_adv_B, W) @ theta_B))
    else:
        X_adv_A = pgd_attack(theta_A, W, X_test, y_test, config.EPSILON, config.PGD_ALPHA, config.PGD_STEPS)
        rob_A   = accuracy_score(y_test, np.sign(get_random_features(X_adv_A, W) @ theta_A))
        X_adv_B = pgd_attack(theta_B, W, X_test, y_test, config.EPSILON, config.PGD_ALPHA, config.PGD_STEPS)
        rob_B   = accuracy_score(y_test, np.sign(get_random_features(X_adv_B, W) @ theta_B))

    res = {
        'N': N_features, 'Lambda': lam_ridge, 'Sigma': sigma_scale, 'Noise': noise_rate, 'Seed': seed,
        'Acc_A': acc_A, 'Rob_A': rob_A, 'Norm_A': norm_A,
        'Acc_B': acc_B, 'Rob_B': rob_B, 'Norm_B': norm_B
    }
    
    if include_lda:
        acc_L = accuracy_score(y_test, np.sign(Phi_test @ theta_L))
        X_adv_L = auto_attack(theta_L, W, X_test, y_test, config.EPSILON, config.AUTO_ATTACK_STEPS, config.AUTO_ATTACK_RESTARTS)
        rob_L = accuracy_score(y_test, np.sign(get_random_features(X_adv_L, W) @ theta_L))
        res.update({'Acc_L': acc_L, 'Rob_L': rob_L, 'Norm_L': np.linalg.norm(theta_L)})
        
    return res

def run_experiment_MNIST():
    """MNIST (0 vs 1): Robustness vs Overparametrization."""
    print("\n>>> RUNNING MNIST EXPERIMENT: Robustness vs Overparam <<<")
    N_values = [64, 256, 512, 1024, 2048]
    X_train, y_train, X_test, y_test = load_mnist_binary(config.N_TRAIN, config.N_TEST, 
                                                       class0=config.MNIST_CLASSES[0], 
                                                       class1=config.MNIST_CLASSES[1])
    
    results = []
    for N in N_values:
        for s in range(config.N_SEEDS):
            W = np.random.randn(N, config.D_MNIST)
            Phi_train = get_random_features(X_train, W)
            Phi_test = get_random_features(X_test, W)

            # Train models
            theta_R = train_ridge(Phi_train, y_train, lam=1e-6)
            theta_G = train_generative(Phi_train, y_train)
            theta_L = train_lda(Phi_train, y_train, lam=1e-2)

            # Robustness evaluation
            X_adv_R = auto_attack(theta_R, W, X_test, y_test, config.EPSILON, config.AUTO_ATTACK_STEPS, config.AUTO_ATTACK_RESTARTS)
            rob_R = accuracy_score(y_test, np.sign(get_random_features(X_adv_R, W) @ theta_R))

            X_adv_G = auto_attack(theta_G, W, X_test, y_test, config.EPSILON, config.AUTO_ATTACK_STEPS, config.AUTO_ATTACK_RESTARTS)
            rob_G = accuracy_score(y_test, np.sign(get_random_features(X_adv_G, W) @ theta_G))

            X_adv_L = auto_attack(theta_L, W, X_test, y_test, config.EPSILON, config.AUTO_ATTACK_STEPS, config.AUTO_ATTACK_RESTARTS)
            rob_L = accuracy_score(y_test, np.sign(get_random_features(X_adv_L, W) @ theta_L))

            results.append({
                'N': N, 'Ratio': N/config.N_TRAIN, 'Seed': s,
                'Rob_R': rob_R, 'Norm_R': np.linalg.norm(theta_R),
                'Rob_G': rob_G, 'Norm_G': np.linalg.norm(theta_G),
                'Rob_L': rob_L, 'Norm_L': np.linalg.norm(theta_L)
            })
        print(f"N={N}: Rob(R/G/L) = {np.mean([r['Rob_R'] for r in results if r['N']==N]):.2f}/"
              f"{np.mean([r['Rob_G'] for r in results if r['N']==N]):.2f}/"
              f"{np.mean([r['Rob_L'] for r in results if r['N']==N]):.2f}")

    return pd.DataFrame(results)

def run_experiment_CIFAR():
    """CIFAR-10 (Cat vs Dog): Robustness vs Overparametrization using Pre-trained features."""
    print("\n>>> RUNNING CIFAR EXPERIMENT: Robustness vs Overparam (ResNet18 Feats) <<<")
    N_values = [64, 256, 512, 1024, 2048]
    X_real, y_real = get_cifar_features(n_samples=config.N_TRAIN, classes=config.CIFAR_CLASSES)
    
    # Add Label Noise to training data
    noise_idx = np.random.choice(len(y_real), int(config.BASE_NOISE * len(y_real)), replace=False)
    y_real_noisy = y_real.copy()
    y_real_noisy[noise_idx] *= -1

    # Evaluation data: Use synthetic for consistency as per user's last cell logic
    # or could use CIFAR test set features. User code used synthetic test data.
    X_eval, y_eval_clean = generate_data(n=config.N_TEST, d=config.D_CIFAR, sigma_scale=1.0, noise=0.0)

    results = []
    for N in N_values:
        for s in range(config.N_SEEDS):
            W = np.random.randn(N, config.D_CIFAR)
            Phi_train = get_random_features(X_real, W)
            Phi_test = get_random_features(X_eval, W)

            theta_R = train_ridge(Phi_train, y_real_noisy, lam=1e-6)
            theta_G = train_generative(Phi_train, y_real_noisy)

            X_adv_R = auto_attack(theta_R, W, X_eval, y_eval_clean, config.EPSILON, config.AUTO_ATTACK_STEPS, config.AUTO_ATTACK_RESTARTS)
            rob_R = accuracy_score(y_eval_clean, np.sign(get_random_features(X_adv_R, W) @ theta_R))

            X_adv_G = auto_attack(theta_G, W, X_eval, y_eval_clean, config.EPSILON, config.AUTO_ATTACK_STEPS, config.AUTO_ATTACK_RESTARTS)
            rob_G = accuracy_score(y_eval_clean, np.sign(get_random_features(X_adv_G, W) @ theta_G))

            results.append({
                'N': N, 'Ratio': N/config.N_TRAIN, 'Seed': s,
                'Rob_R': rob_R, 'Norm_R': np.linalg.norm(theta_R),
                'Rob_G': rob_G, 'Norm_G': np.linalg.norm(theta_G)
            })
        print(f"N={N}: Rob(R/G) = {np.mean([r['Rob_R'] for r in results if r['N']==N]):.3f}/"
              f"{np.mean([r['Rob_G'] for r in results if r['N']==N]):.3f}")

    return pd.DataFrame(results)

def run_experiment_A():
    """Isotropic Data, Varying N (The Curse)"""
    print("\n>>> RUNNING EXPERIMENT A: The Curse (Varying N) <<<")
    n_values = np.unique(np.logspace(np.log10(config.D), np.log10(2000), 10).astype(int))
    results = []
    for N in n_values:
        for s in range(config.N_SEEDS):
            res = run_single_trial(N, 1e-6, 1.0, config.BASE_NOISE, s)
            results.append(res)
        avg_rob_A = np.mean([r['Rob_A'] for r in results if r['N'] == N])
        avg_rob_B = np.mean([r['Rob_B'] for r in results if r['N'] == N])
        print(f"N={N}: Avg Rob_A={avg_rob_A:.2f}, Avg Rob_B={avg_rob_B:.2f}")
    return pd.DataFrame(results)

def run_experiment_B():
    """Regularization Sweep (The Tradeoff)"""
    print("\n>>> RUNNING EXPERIMENT B: The Tradeoff (Varying Lambda) <<<")
    lam_values = np.logspace(-6, 2, 10)
    results = []
    for lam in lam_values:
        for s in range(config.N_SEEDS):
            res = run_single_trial(2000, lam, 1.0, config.BASE_NOISE, s)
            results.append(res)
        avg_acc_A = np.mean([r['Acc_A'] for r in results if r['Lambda'] == lam])
        avg_rob_A = np.mean([r['Rob_A'] for r in results if r['Lambda'] == lam])
        print(f"Lam={lam:.1e}: Avg Acc_A={avg_acc_A:.3f}, Avg Rob_A={avg_rob_A:.3f}")
    
    df = pd.DataFrame(results)
    # Return df and the average generative point
    gen_point = (df['Acc_B'].mean(), df['Rob_B'].mean())
    return df, gen_point

def run_experiment_C():
    """Anisotropic Data (The Hidden Cost)"""
    print("\n>>> RUNNING EXPERIMENT C: Hidden Cost (Anisotropic) <<<")
    n_values = np.unique(np.logspace(np.log10(config.D), np.log10(2000), 10).astype(int))
    results = []
    for N in n_values:
        for s in range(config.N_SEEDS):
            res = run_single_trial(N, 1e-6, 25.0, config.BASE_NOISE, s)
            results.append(res)
        avg_acc_A = np.mean([r['Acc_A'] for r in results if r['N'] == N])
        avg_acc_B = np.mean([r['Acc_B'] for r in results if r['N'] == N])
        print(f"N={N}: Avg Acc_A={avg_acc_A:.2f}, Avg Acc_B={avg_acc_B:.2f}")
    return pd.DataFrame(results)

def run_experiment_D():
    """Noise Sensitivity (Varying Noise Rate)"""
    print("\n>>> RUNNING EXPERIMENT D: Noise Sensitivity (Varying Noise Rate) <<<")
    noise_values = np.linspace(0.0, 0.4, 10)
    results = []
    for nz in noise_values:
        for s in range(config.N_SEEDS):
            res = run_single_trial(2000, 1e-6, 1.0, nz, s)
            results.append(res)
        avg_rob_A = np.mean([r['Rob_A'] for r in results if r['Noise'] == nz])
        avg_rob_B = np.mean([r['Rob_B'] for r in results if r['Noise'] == nz])
        print(f"Noise={nz:.2f}: Avg Rob_A={avg_rob_A:.2f}, Avg Rob_B={avg_rob_B:.2f}")
    return pd.DataFrame(results)

def run_experiment_E():
    """Adversarial Training Comparison (The Defense)"""
    print("\n>>> RUNNING EXPERIMENT E: Adversarial Training Comparison <<<")
    # Focus on the Overparametrized Regime where the Curse happens
    N_values = np.linspace(256, 2048, 6).astype(int)
    results = []

    for s in range(config.N_SEEDS):
        # 1. Data Generation
        X_tr, y_tr = generate_data(config.N_TRAIN, config.D, sigma_scale=1.0, noise=config.BASE_NOISE, seed=s)
        X_te, y_te = generate_data(config.N_TEST, config.D, sigma_scale=1.0, noise=0.0, seed=None)

        for N in N_values:
            W = np.random.randn(N, config.D)
            Phi_tr = get_random_features(X_tr, W)
            Phi_te = get_random_features(X_te, W)

            # --- Model 1: Standard Discriminative (Ridge) ---
            th_std = train_ridge(Phi_tr, y_tr, lam=1e-6)

            # --- Model 2: Generative (Centroid) ---
            th_gen = train_generative(Phi_tr, y_tr)

            # --- Model 3: Adversarial Training (Ridge + AT) ---
            # Step A: Attack the training data using the standard model
            X_tr_adv = pgd_attack(th_std, W, X_tr, y_tr, config.EPSILON, config.PGD_ALPHA, config.PGD_STEPS)
            Phi_tr_adv = get_random_features(X_tr_adv, W)

            # Step B: Train on mixture (Clean + Adversarial)
            Phi_combined = np.vstack([Phi_tr, Phi_tr_adv])
            y_combined = np.hstack([y_tr, y_tr])
            th_at = train_ridge(Phi_combined, y_combined, lam=1e-6)

            # --- EVALUATION (Robust Accuracy on Test Set) ---
            # Attack Standard Model
            X_adv_std = pgd_attack(th_std, W, X_te, y_te, config.EPSILON, config.PGD_ALPHA, config.PGD_STEPS)
            rob_std = accuracy_score(y_te, np.sign(get_random_features(X_adv_std, W) @ th_std))

            # Attack Generative Model
            X_adv_gen = pgd_attack(th_gen, W, X_te, y_te, config.EPSILON, config.PGD_ALPHA, config.PGD_STEPS)
            rob_gen = accuracy_score(y_te, np.sign(get_random_features(X_adv_gen, W) @ th_gen))

            # Attack AT Model
            X_adv_at = pgd_attack(th_at, W, X_te, y_te, config.EPSILON, config.PGD_ALPHA, config.PGD_STEPS)
            rob_at = accuracy_score(y_te, np.sign(get_random_features(X_adv_at, W) @ th_at))

            results.append({
                'Ratio': N/config.N_TRAIN, 'Seed': s, 'N': N,
                'Rob_Standard': rob_std,
                'Rob_Generative': rob_gen,
                'Rob_AT': rob_at
            })
        print(f"Seed {s} Complete. Final N={N}: Std={rob_std:.2f}, Gen={rob_gen:.2f}, AT={rob_at:.2f}")

    return pd.DataFrame(results)
