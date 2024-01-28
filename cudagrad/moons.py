from os import getenv

PROFILING = int(getenv("PROFILING", "0"))

if not PROFILING:
    import matplotlib.pyplot as plt
    import numpy as np

###############################################################################


from random import choice, random

from sklearn.datasets import make_moons

from cudagrad.mlp import MLP
from cudagrad.nn import mse, sgd
from cudagrad.tensor import Tensor

if __name__ == "__main__":
    moons = make_moons()  # two moons

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
            print(f"{epoch=}", f"{loss.item()=}")
            epochs.append(epoch)
            losses.append(loss.item())
            accuracy = []
            for i, target in enumerate(targets):
                accuracy.append(round(model(Tensor([2, 1], inputs[i])).item()) == target.item())
            print(f"{round(sum(accuracy) / len(accuracy), 2)}%")
            print(["🔥" if x == True else " " for x in accuracy])

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
