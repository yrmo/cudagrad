# cudagrad

A small tensor-valued autograd engine, inspired by [PyTorch](https://github.com/pytorch/pytorch) and [micrograd](https://github.com/karpathy/micrograd)

![](https://upload.wikimedia.org/wikipedia/commons/thumb/4/48/Sphyraena_barracuda_%28great_barracuda%29_%28Little_San_Salvador_Island%2C_Bahamas%29_%2816182815352%29.jpg/320px-Sphyraena_barracuda_%28great_barracuda%29_%28Little_San_Salvador_Island%2C_Bahamas%29_%2816182815352%29.jpg)

Great barracuda photo by James St. John, [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/), via [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Sphyraena_barracuda_(great_barracuda)_(Little_San_Salvador_Island,_Bahamas)_(16182815352).jpg)

## Example

Available on [PyPI](https://pypi.org/project/cudagrad/) (`pip install cudagrad`)

```py
# python -m pip install cudagrad; python ./examples/example.py
import cudagrad as cg

a = cg.tensor([2, 2], [2.0, 3.0, 4.0, 5.0])
b = cg.tensor([2, 2], [6.0, 7.0, 8.0, 9.0])
c = cg.tensor([2, 2], [10.0, 10.0, 10.0, 10.0])
d = cg.tensor([2, 2], [11.0, 11.0, 11.0, 11.0])
e = ((a @ b) + c) * d
f = e.sum()
f.backward()

print(f.data)  # [2794.0]
print(f.size)  # [1]
print(a.grad)  # [143.0, 187.0, 143.0, 187.0]
print(b.grad)  # [66.0, 66.0, 88.0, 88.0]
```

WIP! TODO: CUDA operation integration and release on PyPI

## Performance

### Tiny matmul

```
$ python -m timeit -s "import cudagrad as cg;" "a = cg.tensor([2, 2], [2.0, 3.0, 4.0, 5.0]); b = cg.tensor([2, 2], [6.0, 7.0, 8.0, 9.0]); c = a @ b"
200000 loops, best of 5: 1.77 usec per loop
```

```
$ python -m timeit -s "import torch;" "a = torch.tensor(((2.0, 3.0), (4.0, 5.0))); b = torch.tensor(((6.0, 7.0), (8.0, 9.0))); c = a @ b"
50000 loops, best of 5: 5.09 usec per loop
```

### Tiny backward

```
$ python -m timeit -s "import cudagrad as cg;" "a = cg.tensor([2, 2], [2.0, 3.0, 4.0, 5.0]); b = cg.tensor([2, 2], [6.0, 7.0, 8.0, 9.0]); c = a @ b; d = c.sum(); d.backward()"
100000 loops, best of 5: 2.64 usec per loop
```

```
$ python -m timeit -s "import torch;" "a = torch.tensor(((2.0, 3.0), (4.0, 5.0)), requires_grad=True); b = torch.tensor(((6.0, 7.0), (8.0, 9.0)), requires_grad=True); c = a @ b; d = c.sum(); d.backward()"
10000 loops, best of 5: 22.3 usec per loop
```

## License

MIT
