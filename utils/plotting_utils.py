import matplotlib.pyplot as plt

def plot_experiment_a(df_A):
    """Plot for Experiment A: The Curse of Overparametrization."""
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({'font.size': 12, 'lines.linewidth': 2.5})

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax2 = ax1.twinx()

    l1, = ax1.plot(df_A['Ratio'], df_A['Rob_A'], 'r-o', label='Discrim. Robust Acc')
    l2, = ax1.plot(df_A['Ratio'], df_A['Rob_B'], 'b-s', label='Generative Robust Acc')
    l3, = ax2.plot(df_A['Ratio'], df_A['Norm_A'], 'r--', alpha=0.5, label='Discrim. Norm')
    l4, = ax2.plot(df_A['Ratio'], df_A['Norm_B'], 'b--', alpha=0.5, label='Generative Norm')

    ax1.set_xlabel('Overparametrization (N/n)')
    ax1.set_ylabel('Robust Accuracy')
    ax2.set_ylabel('Weight Norm (Log Scale)')
    ax2.set_yscale('log')

    plt.legend([l1, l2, l3, l4], [l.get_label() for l in [l1, l2, l3, l4]], loc='center right')
    plt.title('Exp A: The Curse of Overparametrization')
    plt.tight_layout()
    plt.show()

def plot_experiment_b(df_B, point_B):
    """Plot for Experiment B: Accuracy-Robustness Pareto Frontier."""
    plt.figure(figsize=(7, 5))
    plt.plot(df_B['Acc_A'], df_B['Rob_A'], 'r-o', label='Discriminative (Varying Lambda)')
    plt.scatter([point_B[0]], [point_B[1]], c='blue', s=150, marker='*', label='Generative (No Tuning)', zorder=10)
    plt.xlabel('Standard Accuracy')
    plt.ylabel('Robust Accuracy')
    plt.title('Exp B: Accuracy-Robustness Pareto Frontier')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_experiment_c(df_C, N_TRAIN):
    """Plot for Experiment C: Hidden Cost on Anisotropic Data."""
    plt.figure(figsize=(7, 5))
    plt.plot(df_C['N']/N_TRAIN, df_C['Acc_A'], 'r-o', label='Discriminative (Ridge)')
    plt.plot(df_C['N']/N_TRAIN, df_C['Acc_B'], 'b-s', label='Generative (Centroid)')
    plt.xlabel('Overparametrization (N/n)')
    plt.ylabel('Standard Accuracy')
    plt.title('Exp C: Hidden Cost on Anisotropic Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
