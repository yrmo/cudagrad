from random import random
from itertools import product
from functools import reduce
from operator import mul

from cudagrad.tensor import Tensor  # type: ignore
import matplotlib.pyplot as plt


def mse(predicted: Tensor, actual: Tensor) -> Tensor:
    return (predicted - actual) * (predicted - actual)


class Module:
    def parameters(self):
        return [
            getattr(self, attr)
            for attr in dir(self)
            if type(getattr(self, attr)) == Tensor
        ]

    def zero_grad(self):
        for parameter in self.parameters():
            assert type(parameter) == Tensor
            parameter.zero_grad()


def sgd(model: Module, lr: float) -> None:
    def positions(tensor):
        indices = [list(range(size)) for size in tensor.size]
        for index in product(*indices):
            yield index

    for parameter in model.parameters():
        for position in positions(parameter):
            parameter.data[list(position)] = parameter.data[list(position)].item() + (
                -lr * parameter.grad[reduce(mul, position)]
            )


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

    EPOCHS = 100
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
            print("0 OR 0 = ", out0, "🔥" if out0 == 0 else "🌧️")
            print("0 OR 1 = ", out1, "🔥" if out1 == 1 else "🌧️")
            print("1 OR 0 = ", out2, "🔥" if out2 == 1 else "🌧️")
            print("1 OR 1 = ", out3, "🔥" if out3 == 1 else "🌧️")

plt.scatter(epochs, losses)
plt.title("Neuron trained on binary OR function")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("neuron.jpg")
