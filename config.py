# Experiment Configuration

# Data Settings
D = 64
N_TRAIN = 400
N_TEST = 1000
BASE_NOISE = 0.15 # Default noise for Exp A, B, C

# Attack Settings
EPSILON = 0.2
PGD_STEPS = 7
PGD_ALPHA = 0.05

# Simulation Settings
N_SEEDS = 10      # Number of trials per point
SAVE_DIR = "experiment_logs_v2"

# Real Data Settings
D_MNIST = 784
D_CIFAR = 512 # ResNet18 features
MNIST_CLASSES = (0, 1)
CIFAR_CLASSES = (3, 5) # Cat, Dog

# Attack Settings (Upgraded)
AUTO_ATTACK_STEPS = 10
AUTO_ATTACK_RESTARTS = 3
CERTIFY_SAMPLES = 100
SIGMA_SMOOTH = 0.2
