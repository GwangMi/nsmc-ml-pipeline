import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch_model import MLP
from torchvision import datasets, transforms

DATA_DIR = "../data"


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            X = X.view(X.size(0), -1).to(device)
            y = y.to(device)
            preds = model(X).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.ToTensor()
    train_set = datasets.FashionMNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST(root=DATA_DIR, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

    model = MLP().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.5)
    criterion = nn.CrossEntropyLoss()

    epochs = 10
    for epoch in range(1, epochs + 1):
        model.train()
        for X, y in train_loader:
            X = X.view(X.size(0), -1).to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        train_acc = evaluate(model, train_loader, device)
        test_acc = evaluate(model, test_loader, device)
        print(
            f"epoch {epoch:2d} | loss {loss.item():.4f} | "
            f"train_acc {train_acc:.4f} | test_acc {test_acc:.4f}"
        )


if __name__ == "__main__":
    main()
