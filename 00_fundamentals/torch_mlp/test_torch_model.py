import torch
from torch_model import MLP


def test_forward_shape():
    model = MLP(input_dim=784, hidden_dim=16, output_dim=10)
    X = torch.randn(8, 784)

    logits = model(X)

    assert logits.shape == (8, 10)


def test_backward_reduces_loss():
    torch.manual_seed(0)
    model = MLP(input_dim=20, hidden_dim=8, output_dim=3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    criterion = torch.nn.CrossEntropyLoss()

    X = torch.randn(16, 20)
    y = torch.randint(0, 3, (16,))

    loss_before = criterion(model(X), y).item()
    for _ in range(50):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
    loss_after = criterion(model(X), y).item()

    assert loss_after < loss_before
