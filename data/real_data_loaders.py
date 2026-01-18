import numpy as np
import os

# Optional imports - will fail gracefully if not installed
try:
    from tensorflow.keras.datasets import mnist
except ImportError:
    mnist = None

try:
    import torch
    import torchvision
    import torchvision.transforms as transforms
except ImportError:
    torch = None
    torchvision = None

def load_mnist_binary(n_train, n_test, class0=0, class1=1, seed=None):
    """Load MNIST and filter for two classes, returning normalized flat vectors."""
    if mnist is None:
        raise ImportError("Tensorflow is required for MNIST loading. Please install it.")
        
    if seed is not None: np.random.seed(seed)
    (x_tr, y_tr), (x_te, y_te) = mnist.load_data()

    def filter_data(x, y, n):
        mask = (y == class0) | (y == class1)
        x, y = x[mask], y[mask]
        x = x.reshape(x.shape[0], -1) / 255.0
        x = (x - 0.5) * 2.0 # Normalize -1 to 1
        y = np.where(y == class0, -1, 1)
        # Handle case where n > available samples
        n = min(n, len(y))
        idx = np.random.choice(len(y), n, replace=False)
        return x[idx], y[idx]

    X_train, y_train = filter_data(x_tr, y_tr, n_train)
    X_test, y_test = filter_data(x_te, y_te, n_test)
    return X_train, y_train, X_test, y_test

def get_cifar_features(n_samples=1000, classes=(3, 5), device='cpu'):
    """Extract ResNet18 features from CIFAR-10 for a binary subset."""
    if torch is None or torchvision is None:
        raise ImportError("PyTorch and torchvision are required for CIFAR feature extraction.")

    # 1. Load Pre-trained ResNet
    model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
    model.fc = torch.nn.Identity() # Remove last layer to get 512D features
    model.to(device)
    model.eval()

    # 2. Load Data
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # Note: Downloading to ./data
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # 3. Filter for Binary Class (e.g., Cat=3, Dog=5)
    class0, class1 = classes
    idx = [i for i in range(len(trainset)) if trainset.targets[i] in [class0, class1]]
    subset = torch.utils.data.Subset(trainset, idx[:n_samples])
    loader = torch.utils.data.DataLoader(subset, batch_size=32, shuffle=False)

    # 4. Extract Features
    features = []
    labels = []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            feats = model(imgs).cpu().numpy()
            features.append(feats)
            labels.append(lbls.numpy())

    X = np.vstack(features) # Shape (n, 512)
    y = np.hstack(labels)
    # Convert labels to -1, 1
    y = np.where(y == class0, -1, 1)

    return X, y
