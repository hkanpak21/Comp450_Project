import matplotlib.pyplot as plt
import numpy as np

# Aesthetics
COLOR_DISC_MAIN = "#D62728" # Strong Red
COLOR_DISC_FILL = "#FF9896" # Light Red
COLOR_GEN_MAIN  = "#1F77B4" # Strong Blue
COLOR_GEN_FILL  = "#AEC7E8" # Light Blue
COLOR_LDA_MAIN  = "#2CA02C" # Strong Green
COLOR_LDA_FILL  = "#98DF8A" # Light Green

def plot_with_error_bars(ax, df, x_col, y_col_mean, label, color_main, color_fill, linestyle='-'):
    """Plot mean with shaded area for standard deviation."""
    # Group by the x-axis variable and aggregate
    grouped = df.groupby(x_col)[y_col_mean].agg(['mean', 'std']).reset_index()

    x = grouped[x_col]
    mu = grouped['mean']
    sigma = grouped['std'].fillna(0) # Handle single values

    ax.plot(x, mu, color=color_main, label=label, linewidth=2.5, linestyle=linestyle)
    ax.fill_between(x, mu - sigma, mu + sigma, color=color_fill, alpha=0.3)

def plot_experiment_a(df_A, N_TRAIN):
    """Plot for Experiment A: The Curse of Overparametrization."""
    df_A['Ratio'] = df_A['N'] / N_TRAIN
    
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({'font.size': 12})
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax2 = ax1.twinx()

    plot_with_error_bars(ax1, df_A, 'Ratio', 'Rob_A', 'Discriminative (Ridge)', COLOR_DISC_MAIN, COLOR_DISC_FILL)
    plot_with_error_bars(ax1, df_A, 'Ratio', 'Rob_B', 'Generative (Centroid)', COLOR_GEN_MAIN, COLOR_GEN_FILL)

    # Plot Norms on secondary axis (Dashed)
    gA = df_A.groupby('Ratio')['Norm_A'].mean()
    gB = df_A.groupby('Ratio')['Norm_B'].mean()

    ax2.plot(gA.index, gA, color=COLOR_DISC_MAIN, linestyle='--', alpha=0.6, label='Discrim. Norm')
    ax2.plot(gB.index, gB, color=COLOR_GEN_MAIN, linestyle='--', alpha=0.6, label='Generative Norm')

    ax1.set_xlabel('Overparametrization Ratio ($N/n$)')
    ax1.set_ylabel('Robust Accuracy', fontweight='bold')
    ax2.set_ylabel('Weight Norm $\|\Theta\|_2$', rotation=270, labelpad=20)
    ax2.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower left')
    ax2.legend(loc='center right')
    plt.title('Exp A: The Curse of Overparametrization')
    plt.tight_layout()
    plt.show()

def plot_experiment_b(df_B, point_B):
    """Plot for Experiment B: Accuracy-Robustness Pareto Frontier."""
    plt.figure(figsize=(8, 6))

    # Discrim Curve
    g_B = df_B.groupby('Lambda')[['Acc_A', 'Rob_A']].mean().reset_index()
    plt.plot(g_B['Acc_A'], g_B['Rob_A'], color=COLOR_DISC_MAIN, marker='o', label='Discriminative (Varying $\lambda$)')

    # Generative Point
    gen_acc_mean = df_B['Acc_B'].mean()
    gen_acc_std = df_B['Acc_B'].std()
    gen_rob_mean = df_B['Rob_B'].mean()
    gen_rob_std = df_B['Rob_B'].std()

    plt.errorbar(gen_acc_mean, gen_rob_mean, xerr=gen_acc_std, yerr=gen_rob_std,
                 fmt='o', color=COLOR_GEN_MAIN, markersize=10, label='Generative (Fixed)', capsize=5)

    plt.xlabel('Standard Accuracy')
    plt.ylabel('Robust Accuracy')
    plt.title('Exp B: Accuracy vs. Robustness Trade-off')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_experiment_c(df_C, N_TRAIN):
    """Plot for Experiment C: Hidden Cost on Anisotropic Data."""
    df_C['Ratio'] = df_C['N'] / N_TRAIN
    plt.figure(figsize=(8, 6))
    plot_with_error_bars(plt.gca(), df_C, 'Ratio', 'Acc_A', 'Discriminative (Ridge)', COLOR_DISC_MAIN, COLOR_DISC_FILL)
    plot_with_error_bars(plt.gca(), df_C, 'Ratio', 'Acc_B', 'Generative (Centroid)', COLOR_GEN_MAIN, COLOR_GEN_FILL)
    plt.xlabel('Overparametrization Ratio ($N/n$)')
    plt.ylabel('Standard Accuracy')
    plt.title('Exp C: Performance on Anisotropic Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_experiment_d(df_D):
    """Plot for Experiment D: Noise Sensitivity."""
    plt.figure(figsize=(8, 6))
    plot_with_error_bars(plt.gca(), df_D, 'Noise', 'Rob_A', 'Discriminative (Ridge)', COLOR_DISC_MAIN, COLOR_DISC_FILL)
    plot_with_error_bars(plt.gca(), df_D, 'Noise', 'Rob_B', 'Generative (Centroid)', COLOR_GEN_MAIN, COLOR_GEN_FILL)
    plt.xlabel('Label Noise Rate ($\eta$)')
    plt.ylabel('Robust Accuracy')
    plt.title('Impact of Label Noise on Robustness')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_experiment_mnist(df):
    """Plot for MNIST (0 vs 1): Robustness vs Overparametrization."""
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax2 = ax1.twinx()

    plot_with_error_bars(ax1, df, 'Ratio', 'Rob_R', 'Ridge Robust Acc', COLOR_DISC_MAIN, COLOR_DISC_FILL)
    plot_with_error_bars(ax1, df, 'Ratio', 'Rob_G', 'Generative Robust Acc', COLOR_GEN_MAIN, COLOR_GEN_FILL)
    plot_with_error_bars(ax1, df, 'Ratio', 'Rob_L', 'LDA Robust Acc', COLOR_LDA_MAIN, COLOR_LDA_FILL)

    # Weight Norms on secondary axis (Log Scale)
    gR = df.groupby('Ratio')['Norm_R'].mean()
    gG = df.groupby('Ratio')['Norm_G'].mean()
    gL = df.groupby('Ratio')['Norm_L'].mean()

    ax2.plot(gR.index, gR, linestyle='--', color=COLOR_DISC_MAIN, alpha=0.6, label='Ridge Norm')
    ax2.plot(gG.index, gG, linestyle='--', color=COLOR_GEN_MAIN, alpha=0.6, label='Gen Norm')
    ax2.plot(gL.index, gL, linestyle='--', color=COLOR_LDA_MAIN, alpha=0.6, label='LDA Norm')

    ax1.set_xlabel('Overparametrization Ratio ($N/n$)', fontweight='bold')
    ax1.set_ylabel('Robust Accuracy', fontweight='bold')
    ax2.set_ylabel('Weight Norm $\|\Theta\|_2$', rotation=270, labelpad=20)
    ax2.set_yscale('log')
    
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower left')
    ax2.legend(loc='center right')
    plt.title('MNIST (0 vs 1): Robustness & Norms', fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_experiment_cifar(df):
    """Plot for CIFAR: Robustness vs Overparametrization."""
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax2 = ax1.twinx()

    plot_with_error_bars(ax1, df, 'Ratio', 'Rob_R', 'Ridge Robust Acc', COLOR_DISC_MAIN, COLOR_DISC_FILL)
    plot_with_error_bars(ax1, df, 'Ratio', 'Rob_G', 'Generative Robust Acc', COLOR_GEN_MAIN, COLOR_GEN_FILL)

    # Weight Norms on secondary axis (Log Scale)
    gR = df.groupby('Ratio')['Norm_R'].mean()
    gG = df.groupby('Ratio')['Norm_G'].mean()

    ax2.plot(gR.index, gR, linestyle='--', color=COLOR_DISC_MAIN, alpha=0.6, label='Ridge Norm')
    ax2.plot(gG.index, gG, linestyle='--', color=COLOR_GEN_MAIN, alpha=0.6, label='Gen Norm')

    ax1.set_xlabel('Overparametrization Ratio ($N/n$)', fontweight='bold')
    ax1.set_ylabel('Robust Accuracy', fontweight='bold')
    ax2.set_ylabel('Weight Norm $\|\Theta\|_2$', rotation=270, labelpad=20)
    ax2.set_yscale('log')
    
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower left')
    ax2.legend(loc='center right')
    plt.title('CIFAR: Robustness & Norms vs. Overparametrization', fontweight='bold')
    plt.tight_layout()
    plt.show()
