# python tests/test.py

import cudagrad as cg
import torch

# assert cg.__version__ == '0.0.1'
assert cg.add(1, 2) == 3
assert cg.subtract(1, 2) == -1

def flatten(l):
  out = []
  for item in l:
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

a = cg.tensor([2, 2], [2.0, 3.0, 4.0, 5.0])
b = cg.tensor([2, 2], [6.0, 7.0, 8.0, 9.0])
c = cg.tensor([2, 2], [10.0, 10.0, 10.0, 10.0])
d = cg.tensor([2, 2], [11.0, 11.0, 11.0, 11.0])
e = ((a @ b) + c) * d
f = e.sum()
f.backward()

assert f.data == flatten([ft.data.tolist()]) # this is annoying lol
assert a.data == flatten(at.data.tolist())
assert b.data == flatten(bt.data.tolist())

print('Tests passed!')
