# type: ignore
#
# Copyright 2023 Ryan Moore
#
# So the magic is that there's this relatively simple algorithm called
# backpropagation that takes the error in the output and sends that
# error backwards through the network and computes through all the
# connections how you should change them to improve the behavior, and
# then you change them all a tiny bit and you just keep going with
# another example. And surprisingly that actually works. For many years
# people thought that would just get jammed up — it would get stuck
# somewhere — but no it doesn't, it actually works very well.
#
# Geoffrey Hinton

# %%
from random import random
from typing import *  # type: ignore

from cudagrad.tensor import Tensor  # type: ignore
from .neuron import Module, mse, sgd
import matplotlib.pyplot as plt

class MLP(Module):
    def __init__(self):
        self.w0 = Tensor([10, 2], [random() for _ in range(10 * 2)])
        self.b0 = Tensor([10], [random() for _ in range(10)])
        self.w1 = Tensor([1, 10], [random() for _ in range(1 * 10)])
        self.b1 = Tensor([1], [random() for _ in range(1)])

    def __call__(self, x: Tensor) -> Tensor:
        return (self.w1 @ ((self.w0 @ x) + self.b0)) + self.b1


if __name__ == "__main__":
    # XOR
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    targets = [0, 1, 1, 0]

    EPOCHS = 5000
    lr = 0.001
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
            print("0 OR 0 = ", out0, "🔥" if out0 == 0 else "🌧️")
            print("0 OR 1 = ", out1, "🔥" if out1 == 1 else "🌧️")
            print("1 OR 0 = ", out2, "🔥" if out2 == 1 else "🌧️")
            print("1 OR 1 = ", out3, "🔥" if out3 == 0 else "🌧️")

    plt.scatter(epochs, losses)
    plt.title("MLP trained on binary XOR function")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("mlp.jpg")
