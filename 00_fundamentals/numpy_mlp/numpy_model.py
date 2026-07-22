import numpy as np


class MLP:
    """2-layer MLP (Linear -> ReLU -> Linear -> Softmax) with hand-derived backprop.

    No autograd: forward() caches every intermediate tensor backward() needs,
    mirroring exactly what PyTorch's graph does under the hood.
    """

    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10, seed=0):
        rng = np.random.default_rng(seed)
        w1_scale = np.sqrt(2 / input_dim)
        w2_scale = np.sqrt(2 / hidden_dim)
        self.W1 = rng.normal(0, w1_scale, (input_dim, hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = rng.normal(0, w2_scale, (hidden_dim, output_dim)).astype(np.float32)
        self.b2 = np.zeros(output_dim, dtype=np.float32)

    def forward(self, X):
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2

        shifted = self.z2 - self.z2.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        self.probs = exp / exp.sum(axis=1, keepdims=True)
        return self.probs

    def backward(self, y_true, lr):
        n = y_true.shape[0]
        y_onehot = np.zeros_like(self.probs)
        y_onehot[np.arange(n), y_true] = 1

        # d(cross_entropy + softmax)/dz2 simplifies to (probs - onehot) / n
        dz2 = (self.probs - y_onehot) / n
        dW2 = self.a1.T @ dz2
        db2 = dz2.sum(axis=0)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self.z1 > 0)
        dW1 = self.X.T @ dz1
        db1 = dz1.sum(axis=0)

        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

    def loss(self, y_true):
        n = y_true.shape[0]
        log_likelihood = -np.log(self.probs[np.arange(n), y_true] + 1e-9)
        return log_likelihood.mean()

    def predict(self, X):
        return self.forward(X).argmax(axis=1)
