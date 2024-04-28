import urllib.request
from os.path import isfile

import numpy as np
from torch import nn, tensor, zeros

filename = "mnist.npz"
if not isfile(filename):
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
        filename,
    )

with np.load(filename, allow_pickle=True) as data:
    train_images = data["x_train"]
    train_labels = data["y_train"]
    test_images = data["x_test"]
    test_labels = data["y_test"]


class ZeroNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return zeros(x.shape)


model = ZeroNet()

for train_image in train_images:
    model(tensor(train_image))
    break
