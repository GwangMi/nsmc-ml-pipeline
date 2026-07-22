import numpy as np
from numpy_model import MLP


def test_forward_shape():
    model = MLP(input_dim=784, hidden_dim=16, output_dim=10, seed=0)
    X = np.random.default_rng(0).normal(size=(8, 784)).astype(np.float32)

    probs = model.forward(X)

    assert probs.shape == (8, 10)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_backward_reduces_loss():
    model = MLP(input_dim=20, hidden_dim=8, output_dim=3, seed=0)
    rng = np.random.default_rng(1)
    X = rng.normal(size=(16, 20)).astype(np.float32)
    y = rng.integers(0, 3, size=16)

    model.forward(X)
    loss_before = model.loss(y)

    for _ in range(50):
        model.forward(X)
        model.backward(y, lr=0.5)
    model.forward(X)
    loss_after = model.loss(y)

    assert loss_after < loss_before
