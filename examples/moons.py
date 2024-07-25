from os import getenv
from random import choice, random

from sklearn.datasets import make_moons

from cudagrad.nn import Module, mse, sgd
from cudagrad.tensor import Tensor

PROFILING = int(getenv("PROFILING", "0"))

if not PROFILING:
    import matplotlib.pyplot as plt
    import numpy as np


class MLP(Module):
    def __init__(self) -> None:
        self.w0 = Tensor(
            [16, 2], [choice([-1 * random(), random()]) for _ in range(2 * 16)]
        )
        self.b0 = Tensor(
            [16, 1], [choice([-1 * random(), random()]) for _ in range(16)]
        )
        self.w1 = Tensor(
            [1, 16], [choice([-1 * random(), random()]) for _ in range(1 * 16)]
        )
        self.b1 = Tensor([1], [choice([-1 * random(), random()]) for _ in range(1)])

    def __call__(self, x: Tensor) -> Tensor:
        return Tensor.sigmoid(
            self.w1 @ Tensor.sigmoid((self.w0 @ x + self.b0)) + self.b1
        )


if __name__ == "__main__":
    moons = make_moons()  # two moons

    inputs = moons[0]
    targets = moons[1]

    EPOCHS = 500
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
                accuracy.append(
                    round(model(Tensor([2, 1], inputs[i])).item()) == target.item()
                )
            print(f"{round(sum(accuracy) / len(accuracy), 2) * 100}%")
            print("".join(["🔥" if x else " " for x in accuracy]))

    if not PROFILING:
        x = np.linspace(-2.5, 2.5, 50)
        y = np.linspace(-1.5, 1.5, 50)
        X, Y = np.meshgrid(x, y)  # type: ignore [no-untyped-call]
        Z = np.zeros(X.shape)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                input_data = Tensor([2, 1], [X[i, j], Y[i, j]])
                Z[i, j] = model(input_data).item()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        surf = ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.6) # type: ignore [attr-defined]

        ax.scatter(
            inputs[targets == 0, 0], inputs[targets == 0, 1], 0, c="#440154", label="0"
        )
        ax.scatter(
            inputs[targets == 1, 0], inputs[targets == 1, 1], 0, c="#fde725", label="1"
        )

        ax.set_xlabel("X") # type: ignore [attr-defined]
        ax.set_ylabel("Y") # type: ignore [attr-defined]
        ax.set_zlabel("Z") # type: ignore [attr-defined]

        plt.savefig("./examples/plots/moons-3d.jpg")
