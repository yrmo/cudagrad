# TEST BACKWARD MLP

# OrderedDict([('layer1.weight',
#               tensor([[-0.5963, -0.0062],
#                       [ 0.1741, -0.1097]])),
#              ('layer1.bias', tensor([-0.4237, -0.6666])),
#              ('layer2.weight', tensor([[0.1204, 0.2781]])),
#              ('layer2.bias', tensor([-0.4580]))])

# After 4 XOR steps (1 sample each):

# layer1.weight tensor([[0.0054, 0.0054],
#         [0.0140, 0.0140]])
# layer1.bias tensor([0.0054, 0.0140])
# layer2.weight tensor([[0.0556, 0.0747]])
# layer2.bias tensor([0.2105])

# %%

from cudagrad.nn import mse, sgd

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

print(model.w0)
print(model.b0)
print(model.w1)
print(model.b1)

X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
y = [[0.0], [1.0], [1.0], [0.0]]


from itertools import product

def positions(tensor):
    indices = [list(range(size)) for size in tensor.size]
    for index in product(*indices):
        yield index


for i in range(4):
    model.zero_grad()
    loss = mse(Tensor([1], y[i]), model(Tensor([2, 1], X[i])))
    loss.backward()

    print("*" * 80)
    print(f"{X[i]=}", "->", f"{y[i]=}")
    print("w0" + "-" * 20)
    print(model.w0)
    print("b0" + "-" * 20)
    print(model.b0)
    print("w1" + "-" * 20)
    print(model.w1)
    print("b1" + "-" * 20)
    print(model.b1)

    # for position in positions(p):
    #     print(list(position))
    #     print(p.grad[list(position)].item())
    print("-" * 80)

# %%

import torch
import torch.nn as nn
import torch.optim as optim

X = torch.tensor(
    [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], requires_grad=True
)
y = torch.tensor([[0.0], [1.0], [1.0], [0.0]], requires_grad=False)

torch.manual_seed(1337)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 2)
        self.layer2 = nn.Linear(2, 1)

    def forward(self, x):
        x = torch.sigmoid(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        return x


model = MLP()

model.state_dict()

# optimizer = optim.SGD(model.parameters(), lr=0.1)
criterion = nn.MSELoss()

for i in range(4):
    outputs = model(X[i])
    loss = criterion(outputs, y[i])
    # optimizer.zero_grad()
    model.zero_grad()
    loss.backward()
    # optimizer.step()

    # print(f"{model.layer1.weight.data=}")
    print(f"{X[i]=}", "->", f"{y[i]=}")
    print(f"{model.layer1.weight.grad}")
    print(f"{model.layer1.bias.grad}")
    print(f"{model.layer2.weight.grad}")
    print(f"{model.layer2.bias.grad}")
# %%
