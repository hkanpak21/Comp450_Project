from experiments import run_experiment_A, run_experiment_B, run_experiment_C
from utils.plotting_utils import plot_experiment_a, plot_experiment_b, plot_experiment_c
import config

def main():
    # Run Experiments
    df_A = run_experiment_A()
    df_B, point_B = run_experiment_B()
    df_C = run_experiment_C()

    # Visualization
    plot_experiment_a(df_A)
    plot_experiment_b(df_B, point_B)
    plot_experiment_c(df_C, config.N_TRAIN)

    # Print Summary Tables
    print("\n" + "="*40)
    print("RESULTS SUMMARY")
    print("="*40)
    print("\n--- EXPERIMENT A ---")
    print(df_A[['N', 'Ratio', 'Rob_A', 'Norm_A', 'Rob_B', 'Norm_B']].to_string(index=False))
    print("\n--- EXPERIMENT B ---")
    print(f"Generative Point (Acc, Rob): {point_B}")
    print(df_B.to_string(index=False))
    print("\n--- EXPERIMENT C ---")
    print(df_C.to_string(index=False))

if __name__ == "__main__":
    main()
