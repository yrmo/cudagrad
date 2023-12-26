# https://www.youtube.com/watch?v=QrzApibhohY

import torch
import torch.nn as nn
import torch.optim as optim

import cudagrad.mlp
from cudagrad import Tensor

def torch_mlp():
    X = torch.tensor(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], requires_grad=True
    )
    y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], requires_grad=False)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(1337)
            self.layer1 = nn.Linear(2, 2)
            self.layer2 = nn.Linear(2, 1)

        def forward(self, x):
            x = torch.sigmoid(self.layer1(x))
            x = torch.sigmoid(self.layer2(x))
            return x

    def flatten(l):
        ans = []
        for x in l:
            if isinstance(x, (tuple, list)):
                ans.extend(flatten(x))
            else:
                ans.append(x)
        return ans

    print(model.state_dict())
    print(len(flatten([model.state_dict()[key].tolist() for key in model.state_dict()])))

model = cudagrad.mlp.MLP()
PARAMS = [-0.5963, -0.0062, 0.1741, -0.1097, -0.4237, -0.6666, 0.1204, 0.2781, -0.4580]
EPSILON = 1e-7

plus = []
for i in range(9):
    param = PARAMS
    param[i] += EPSILON
    model.w0.data[[0, 0]] = param[0]
    model.w0.data[[0, 1]] = param[1]
    model.w0.data[[1, 0]] = param[2]
    model.w0.data[[1, 1]] = param[3]
    model.b0.data[[0, 0]] = param[4]
    model.b0.data[[1, 0]] = param[5]
    model.w1.data[[0, 0]] = param[6]
    model.w1.data[[0, 1]] = param[7]
    model.b1.data[[0, 0]] = param[8]
    plus.append(model(Tensor([2, 1], [1, 1])).item())

minus = []
for i in range(9):
    param = PARAMS
    param[i] -= EPSILON
    model.w0.data[[0, 0]] = param[0]
    model.w0.data[[0, 1]] = param[1]
    model.w0.data[[1, 0]] = param[2]
    model.w0.data[[1, 1]] = param[3]
    model.b0.data[[0, 0]] = param[4]
    model.b0.data[[1, 0]] = param[5]
    model.w1.data[[0, 0]] = param[6]
    model.w1.data[[0, 1]] = param[7]
    model.b1.data[[0, 0]] = param[8]
    minus.append(model(Tensor([2, 1], [1, 1])).item())

from numpy import array

print((array(plus) - array(minus)) / (2 * EPSILON))