import urllib.request
from os.path import isfile
from random import random

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from cudagrad import Module, Tensor

filename = "mnist.npz"
if not isfile(filename):
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
        filename,
    )

with np.load(filename, allow_pickle=True) as data:  # type: ignore [no-untyped-call]
    train_images = data["x_train"]
    train_labels = data["y_train"]
    test_images = data["x_test"]
    test_labels = data["y_test"]


class Model(Module):
    def __init__(self, inputs: int, outputs: int):
        self.inputs = inputs
        self.outputs = outputs
        self.w = Tensor([outputs, inputs], [random() for _ in range(outputs * inputs)])
        self.b = Tensor([outputs], [random() for _ in range(outputs)])

    def __call__(self, arr: NDArray) -> Tensor:
        assert len(arr.flatten().tolist()) == 784
        x = Tensor([self.inputs, self.outputs], arr.flatten().tolist())
        return (self.w @ x) + self.b


model = Model(784, 1)


def accuracy() -> float:
    outputs = []
    for i, test_image in enumerate(test_images):
        outputs.append(int(model(test_image).item()))

    targets = test_labels.flatten().tolist()
    return (
        (Tensor([len(outputs)], outputs) == Tensor([len(targets)], targets))
        .sum()
        .item()
        / len(targets)
    ) * 100


for i, train_image in enumerate(train_images):
    if i % (len(train_images) // 10) == 0:
        a = accuracy()
        print(f"{a}%", " " if a < 98 else "🔥")

num_row = 3
num_col = 4
fig, axes = plt.subplots(num_row, num_col)
for i in range(num_row * num_col):
    ax = axes[i // num_col, i % num_col]
    ax.imshow(test_images[i], cmap="viridis")
    output = int(model(train_images[i]).item())
    ax.set_title(f"Output: {output}")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

fig.tight_layout()
plt.savefig("./benchmarks/_cudagrad/plots/mnist.jpg")
