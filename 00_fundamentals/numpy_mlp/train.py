import time

from data import iterate_minibatches, load_fashion_mnist
from numpy_model import MLP

DATA_DIR = "../data"


def evaluate(model, X, y):
    preds = model.predict(X)
    return (preds == y).mean()


def main():
    X_train, y_train, X_test, y_test = load_fashion_mnist(DATA_DIR)
    model = MLP(input_dim=784, hidden_dim=128, output_dim=10, seed=0)

    epochs = 10
    batch_size = 128
    lr = 0.5

    for epoch in range(1, epochs + 1):
        start = time.time()
        for X_batch, y_batch in iterate_minibatches(X_train, y_train, batch_size, seed=epoch):
            model.forward(X_batch)
            model.backward(y_batch, lr)

        model.forward(X_train)
        train_loss = model.loss(y_train)
        train_acc = evaluate(model, X_train, y_train)
        test_acc = evaluate(model, X_test, y_test)
        elapsed = time.time() - start
        print(
            f"epoch {epoch:2d} | loss {train_loss:.4f} | "
            f"train_acc {train_acc:.4f} | test_acc {test_acc:.4f} | {elapsed:.1f}s"
        )


if __name__ == "__main__":
    main()
