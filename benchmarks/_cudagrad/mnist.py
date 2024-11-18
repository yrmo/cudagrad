import urllib.request
from os import getenv
from os.path import isfile
from random import random

import numpy as np
from numpy.typing import NDArray

from cudagrad import Module, Tensor
from cudagrad.nn import cross_entropy, sgd

PROFILING = int(getenv("PROFILING", "0"))

if not PROFILING:
    import matplotlib.pyplot as plt

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
        mean = np.mean(arr)
        std = np.std(arr)
        normalized = (arr - mean) / std
        assert len(normalized.flatten().tolist()) == 784
        x = Tensor([self.inputs, 1], arr.flatten().tolist())
        # should be x @ W.T
        print(self.w.size)
        print(x.size)
        return (self.w @ x) + self.b


model = Model(784, 10)


def accuracy() -> float:
    outputs = []
    for i, test_image in enumerate(test_images):
        x = model(test_image)
        m = x.max().item()
        for index in range(10):
            if x.data[0, index].item() == m:
                break 
        outputs.append(int(index))

    print(outputs)
    targets = test_labels.flatten().tolist()
    return (
        (Tensor([len(outputs)], outputs) == Tensor([len(targets)], targets))
        .sum()
        .item()
        / len(targets)
    ) * 100


for i, train_image in enumerate(train_images):
    model.zero_grad()
    output = model(train_image)
    arr = [output.data[0, 0].item(), output.data[0, 1].item(),
                              output.data[0, 2].item(), output.data[0, 3].item(),
                              output.data[0, 4].item(), output.data[0, 5].item(),
                              output.data[0, 6].item(), output.data[0, 7].item(),
                              output.data[0, 8].item(), output.data[0, 9].item()]
    import numpy as np
    normalized = (arr - np.mean(arr)) / np.std(arr)
    print(normalized.tolist())
    output = Tensor([1, 10], normalized.tolist())
    print(output)
    print(output.size)
    target = Tensor([1], [train_labels[i]])
    print(target)
    print(target.size)
    loss = cross_entropy(output, target)
    print(loss)
    loss.backward()
    exit()
    # sgd(model, 0.0001)

    if i % (len(train_images) // 10) == 0:
        print("E: ", 0, "I: ", i, "L: ", loss.item())
        a = accuracy()
        print(f"{a}%", " " if a < 98 else "🔥")

if not PROFILING:
    num_row = 3
    num_col = 4
    fig, axes = plt.subplots(num_row, num_col)
    for i in range(num_row * num_col):
        ax = axes[i // num_col, i % num_col]
        ax.imshow(test_images[i], cmap="viridis")
        x = model(test_images[i])
        output = int(x.data[0, 0].item())
        ax.set_title(f"Output: {output}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    fig.tight_layout()
    plt.savefig("./benchmarks/_cudagrad/plots/mnist.jpg")
