from os import getenv

PROFILING = int(getenv("PROFILING", "0"))

if not PROFILING:
    import matplotlib.pyplot as plt
    import numpy as np

###############################################################################


from random import choice, random

from sklearn.datasets import make_moons

from cudagrad.nn import mse, sgd
from cudagrad.mlp import MLP
from cudagrad.tensor import Tensor


if __name__ == "__main__":
    moons = make_moons() # two moons

    inputs = moons[0]
    targets = moons[1]

    EPOCHS = 10000
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
            out0 = int(model(Tensor([2, 1], inputs[(len(inputs) // 4) * 0])).item())
            out1 = int(model(Tensor([2, 1], inputs[(len(inputs) // 4) * 1])).item())
            out2 = int(model(Tensor([2, 1], inputs[(len(inputs) // 4) * 2])).item())
            out3 = int(model(Tensor([2, 1], inputs[(len(inputs) // 4) * 3])).item())
            print(
                f"Moons {inputs[(len(inputs) // 4) * 0]} = {(targets[i] // 4) * 0}",
                int(model(Tensor([2, 1], inputs[(len(inputs) // 4) * 0])).item()),
                "🔥" if out0 == targets[(len(inputs) // 4) * 0] else "",
            )
            print(
                f"Moons {inputs[(len(inputs) // 4) * 1]} = {(targets[i] // 4) * 1}",
                int(model(Tensor([2, 1], inputs[(len(inputs) // 4) * 1])).item()),
                "🔥" if out1 == targets[(len(inputs) // 4) * 1] else "",
            )
            print(
                f"Moons {inputs[(len(inputs) // 4) * 2]} = {(targets[i] // 4) * 2}",
                int(model(Tensor([2, 1], inputs[(len(inputs) // 4) * 2])).item()),
                "🔥" if out2 == targets[(len(inputs) // 4) * 2] else "",
            )
            print(
                f"Moons {inputs[(len(inputs) // 4) * 3]} = {(targets[i] // 4) * 3}",
                int(model(Tensor([2, 1], inputs[(len(inputs) // 4) * 3])).item()),
                "🔥" if out3 == targets[(len(inputs) // 4) * 3] else "",
            )

    if not PROFILING:
        plt.scatter(epochs, losses)
        plt.title("MLP trained on two moons")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig("./cudagrad/plots/moons.jpg")

        x = np.linspace(-2.5, 2.5, 50)
        y = np.linspace(-2.5, 2.5, 50)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros(X.shape)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                input_data = Tensor([2, 1], [X[i, j], Y[i, j]])
                Z[i, j] = model(input_data).item()

        plt.figure()
        plt.contourf(X, Y, Z, cmap="viridis")
        plt.title("Two Moons MLP Visualization (2D)")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.colorbar()  # adds a color bar to indicate the Z value
        plt.savefig("./cudagrad/plots/moons-2d.jpg")
