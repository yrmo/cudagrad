import urllib.request
from os.path import isfile

import numpy as np
from cudagrad import Tensor

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


class ZeroNet:
    def __init__(self):
        pass

    def __call__(self, x):
        return Tensor([1], [42]).zeros(x)


model = ZeroNet()

for train_image in train_images:
    print(model(train_image.shape))
    break
