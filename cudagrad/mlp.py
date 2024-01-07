from random import choice, random
from typing import *  # type: ignore

import matplotlib.pyplot as plt

from cudagrad.nn import Module, mse, sgd
from cudagrad.tensor import Tensor


class MLP(Module):
    def __init__(self):
        self.w0 = Tensor(
            [2, 2], [choice([-1 * random(), random()]) for _ in range(2 * 2)]
        )
        self.b0 = Tensor([2, 1], [choice([-1 * random(), random()]) for _ in range(2)])
        self.w1 = Tensor(
            [1, 2], [choice([-1 * random(), random()]) for _ in range(1 * 2)]
        )
        self.b1 = Tensor([1], [choice([-1 * random(), random()]) for _ in range(1)])

    def __call__(self, x: Tensor) -> Tensor:
        return Tensor.sigmoid(
            self.w1 @ Tensor.sigmoid((self.w0 @ x + self.b0)) + self.b1
        )


if __name__ == "__main__":
    # XOR
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    targets = [0, 1, 1, 0]

    EPOCHS = 5000
    lr = 0.05
    epochs = []
    losses = []
    model = MLP()
    for epoch in range(EPOCHS + 1):
        for i, input in enumerate(inputs):
            model.zero_grad()
            loss = mse(Tensor([1], [targets[i]]), model(Tensor([2, 1], input)))
            loss.backward()
            sgd(model, lr)
        if epoch % (EPOCHS // 10) == 0:
            print(f"{epoch=}", f"{loss.item()}")
            epochs.append(epoch)
            losses.append(loss.item())
            out0 = round(model(Tensor([2, 1], inputs[0])).item())
            out1 = round(model(Tensor([2, 1], inputs[1])).item())
            out2 = round(model(Tensor([2, 1], inputs[2])).item())
            out3 = round(model(Tensor([2, 1], inputs[3])).item())
            print(
                "0 OR 0 = ",
                round(model(Tensor([2, 1], inputs[0])).item(), 2),
                "🔥" if out0 == 0 else "",
            )
            print(
                "0 OR 1 = ",
                round(model(Tensor([2, 1], inputs[1])).item(), 2),
                "🔥" if out1 == 1 else "",
            )
            print(
                "1 OR 0 = ",
                round(model(Tensor([2, 1], inputs[2])).item(), 2),
                "🔥" if out2 == 1 else "",
            )
            print(
                "1 OR 1 = ",
                round(model(Tensor([2, 1], inputs[3])).item(), 2),
                "🔥" if out3 == 0 else "",
            )

    plt.scatter(epochs, losses)
    plt.title("MLP trained on binary XOR function")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("./cudagrad/plots/mlp.jpg")

    import numpy as np  # type: ignore

    x = np.linspace(0, 1, 50)
    y = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            input_data = Tensor([2, 1], [X[i, j], Y[i, j]])
            Z[i, j] = model(input_data).item()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis")
    plt.title("XOR MLP Visualization")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig("./cudagrad/plots/mlp-3d.jpg")
