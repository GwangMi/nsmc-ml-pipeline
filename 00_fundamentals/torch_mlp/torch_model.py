from torch import nn


class MLP(nn.Module):
    """Same architecture as numpy_mlp/numpy_model.py: Linear -> ReLU -> Linear.

    Softmax + cross-entropy is folded into nn.CrossEntropyLoss in train.py,
    and backward() is handled entirely by autograd.
    """

    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)
