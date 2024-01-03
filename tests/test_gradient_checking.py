from itertools import product

from cudagrad.mlp import MLP
from cudagrad.nn import mse, sgd
from cudagrad.tensor import Tensor


def leet_mlp():
    model = MLP()

    # matches torch 1337 seed
    model.w0.data[[0, 0]] = -0.5963
    model.w0.data[[0, 1]] = -0.0062
    model.w0.data[[1, 0]] = 0.1741
    model.w0.data[[1, 1]] = -0.1097

    model.b0.data[[0, 0]] = -0.4237
    model.b0.data[[1, 0]] = -0.6666

    model.w1.data[[0, 0]] = 0.1204
    model.w1.data[[0, 1]] = 0.2781

    model.b1.data[[0, 0]] = -0.4580

    return model


model = leet_mlp()
x = [1.0, 1.0]
y = 0.0


def positions(tensor):
    indices = [list(range(size)) for size in tensor.size]
    for index in product(*indices):
        yield index


epsilon = 0.1
out = model(Tensor([2, 1], x))
out.backward()

model.w0.data[[0, 0]] = model.w0.data[[0, 0]].item() + epsilon
plus_case = model(Tensor([2, 1], x)).item()

# double minus because cancel out plus epsilon
model.w0.data[[0, 0]] = model.w0.data[[0, 0]].item() - (2 * epsilon)
negative_case = model(Tensor([2, 1], x)).item()

check_grad = (plus_case - negative_case) / (2 * epsilon)

model = leet_mlp()
model.zero_grad()
loss = mse(Tensor([1], [y]), model(Tensor([2, 1], x)))
loss.backward()
backward_grad = model.w0.data[[0, 0]].item()

print(f"{check_grad=}", f"{backward_grad=}")

# print("*" * 80)
# print(f"{x=}", "->", f"{y=}")
# print("w0" + "-" * 20)
# print(model.w0)
# print("b0" + "-" * 20)
# print(model.b0)
# print("w1" + "-" * 20)
# print(model.w1)
# print("b1" + "-" * 20)
# print(model.b1)
