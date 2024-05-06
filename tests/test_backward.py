# python tests/test_backward.py

import torch

from cudagrad import Tensor

# assert cg.__version__ == '0.0.1'
# assert cg.add(1, 2) == 3
# assert cg.subtract(1, 2) == -1


def flatten(iterable):
    out = []
    for item in iterable:
        if isinstance(item, (list, tuple)):
            out.extend(flatten(item))
        else:
            out.append(item)
    return out


at = torch.tensor(((2.0, 3.0), (4.0, 5.0)), requires_grad=True)
bt = torch.tensor(((6.0, 7.0), (8.0, 9.0)), requires_grad=True)
ct = torch.tensor(((10.0, 10.0), (10.0, 10.0)), requires_grad=True)
dt = torch.tensor(((11.0, 11.0), (11.0, 11.0)), requires_grad=True)
et = (at.matmul(bt) + ct) * dt
ft = et.sum()
ft.backward()

a = Tensor([2, 2], [2.0, 3.0, 4.0, 5.0])
b = Tensor([2, 2], [6.0, 7.0, 8.0, 9.0])
c = Tensor([2, 2], [10.0, 10.0, 10.0, 10.0])
d = Tensor([2, 2], [11.0, 11.0, 11.0, 11.0])
e = ((a @ b) + c) * d
f = e.sum()
f.backward()

assert at.grad is not None
assert bt.grad is not None

assert f.data[[0, 0]].item() == ft.data.item()

assert [
    a.grad[[0, 0]].item(),
    a.grad[[0, 1]].item(),
    a.grad[[1, 0]].item(),
    a.grad[[1, 1]].item(),
] == flatten(at.grad.tolist())
assert [
    b.grad[[0, 0]].item(),
    b.grad[[0, 1]].item(),
    b.grad[[1, 0]].item(),
    b.grad[[1, 1]].item(),
] == flatten(bt.grad.tolist())
