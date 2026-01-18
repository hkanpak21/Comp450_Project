import config
from experiments import (run_experiment_A, run_experiment_B, run_experiment_C, 
                         run_experiment_D, run_experiment_E, run_experiment_MNIST, run_experiment_CIFAR)
from utils.plotting_utils import (plot_experiment_a, plot_experiment_b, plot_experiment_c, 
                                 plot_experiment_d, plot_experiment_e, plot_experiment_mnist, plot_experiment_cifar)
import pandas as pd

def main():
    # --- 1. Synthetic Data Experiments ---
    df_A = run_experiment_A()
    df_B, point_B = run_experiment_B()
    df_C = run_experiment_C()
    df_D = run_experiment_D()
    df_E = run_experiment_E()

    # Plot Synthetic
    plot_experiment_a(df_A, config.N_TRAIN)
    plot_experiment_b(df_B, point_B)
    plot_experiment_c(df_C, config.N_TRAIN)
    plot_experiment_d(df_D)
    plot_experiment_e(df_E)

    # --- 2. Real Data Experiments ---
    try:
        df_MNIST = run_experiment_MNIST()
        plot_experiment_mnist(df_MNIST)
    except Exception as e:
        print(f"\nSkipping MNIST Experiment: {e}")

    try:
        df_CIFAR = run_experiment_CIFAR()
        plot_experiment_cifar(df_CIFAR)
    except Exception as e:
        print(f"\nSkipping CIFAR Experiment: {e}")

    # --- Final Results Export ---
    print("\n" + "="*40)
    print("RESULTS SUMMARY")
    print("="*40)
    
    print("\n--- EXP A: The Curse ---")
    print(df_A.groupby('N')[['Rob_A', 'Rob_B']].mean().to_string())
    
    print("\n--- EXP D: Noise Sensitivity ---")
    print(df_D.groupby('Noise')[['Rob_A', 'Rob_B']].mean().to_string())

    print("\n--- EXP E: Adversarial Training ---")
    print(df_E.groupby('N')[['Rob_Standard', 'Rob_AT', 'Rob_Generative']].mean().to_string())

    if 'df_MNIST' in locals():
        print("\n--- EXP MNIST ---")
        print(df_MNIST.groupby('N')[['Rob_R', 'Rob_G', 'Rob_L']].mean().to_string())

    if 'df_CIFAR' in locals():
        print("\n--- EXP CIFAR ---")
        print(df_CIFAR.groupby('N')[['Rob_R', 'Rob_G']].mean().to_string())

if __name__ == "__main__":
    main()
