import numpy as np


def load_fashion_mnist(data_dir):
    from torchvision import datasets

    train = datasets.FashionMNIST(root=data_dir, train=True, download=True)
    test = datasets.FashionMNIST(root=data_dir, train=False, download=True)

    X_train = train.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_train = train.targets.numpy()
    X_test = test.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_test = test.targets.numpy()
    return X_train, y_train, X_test, y_test


def iterate_minibatches(X, y, batch_size, shuffle=True, seed=None):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = np.arange(n)
    if shuffle:
        rng.shuffle(idx)
    for start in range(0, n, batch_size):
        batch_idx = idx[start:start + batch_size]
        yield X[batch_idx], y[batch_idx]
