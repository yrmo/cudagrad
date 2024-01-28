from os import getenv

PROFILING = int(getenv("PROFILING", "0"))

if not PROFILING:
    import matplotlib.pyplot as plt
    import numpy as np

###############################################################################

from random import random

from cudagrad.nn import Module, mse, sgd
from cudagrad.tensor import Tensor


class Linear(Module):
    def __init__(self, inputs: int, outputs: int):
        self.w = Tensor([outputs, inputs], [random() for _ in range(outputs * inputs)])
        self.b = Tensor([outputs], [random() for _ in range(outputs)])

    def __call__(self, x: Tensor) -> Tensor:
        return (self.w @ x) + self.b


if __name__ == "__main__":
    # OR
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    targets = [0, 1, 1, 1]

    EPOCHS = 1000
    lr = 0.0001
    epochs = []
    losses = []
    model = Linear(2, 1)
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
                "🔥" if out3 == 1 else "",
            )

    if not PROFILING:
        x1_values = np.linspace(0, 1, 100)
        x2_values = np.linspace(0, 1, 100)
        x1_grid, x2_grid = np.meshgrid(x1_values, x2_values)

        outputs = np.zeros(x1_grid.shape)
        for i in range(x1_grid.shape[0]):
            for j in range(x1_grid.shape[1]):
                outputs[i, j] = model(Tensor([2, 1], [x1_grid[i, j], x2_grid[i, j]])).item()

        plt.imshow(outputs, extent=(0, 1, 0, 1), origin='lower', cmap="viridis", alpha=0.7)
        plt.colorbar(label='Z')
        plt.title('Neuron train on binary OR function')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig("./cudagrad/plots/linear-2d-contour.jpg")
