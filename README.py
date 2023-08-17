from subprocess import run

echo = lambda x : run(x, shell=True, capture_output=True).stdout.decode('utf-8').strip()

cmd1 = 'python -m timeit -s "import cudagrad as cg;" "a = cg.tensor([2, 2], [2.0, 3.0, 4.0, 5.0]); b = cg.tensor([2, 2], [6.0, 7.0, 8.0, 9.0]); c = a @ b"'
cmd2 = 'python -m timeit -s "import torch;" "a = torch.tensor(((2.0, 3.0), (4.0, 5.0))); b = torch.tensor(((6.0, 7.0), (8.0, 9.0))); c = a @ b"'
cmd3 = 'python -m timeit -s "import cudagrad as cg;" "a = cg.tensor([2, 2], [2.0, 3.0, 4.0, 5.0]); b = cg.tensor([2, 2], [6.0, 7.0, 8.0, 9.0]); c = a @ b; d = c.sum(); d.backward()"'
cmd4 = 'python -m timeit -s "import torch;" "a = torch.tensor(((2.0, 3.0), (4.0, 5.0)), requires_grad=True); b = torch.tensor(((6.0, 7.0), (8.0, 9.0)), requires_grad=True); c = a @ b; d = c.sum(); d.backward()"'
cmd5 = 'python -m timeit -s "import numpy as np" "a = np.array([[2.0, 3.0],[4.0, 5.0]]); b = np.array([[6.0, 7.0], [8.0, 9.0]]); c = a @ b;"'

README = f"""\
# cudagrad

A small tensor-valued autograd engine, inspired by [PyTorch](https://github.com/pytorch/pytorch) and [micrograd](https://github.com/karpathy/micrograd)

![](https://upload.wikimedia.org/wikipedia/commons/thumb/4/48/Sphyraena_barracuda_%28great_barracuda%29_%28Little_San_Salvador_Island%2C_Bahamas%29_%2816182815352%29.jpg/320px-Sphyraena_barracuda_%28great_barracuda%29_%28Little_San_Salvador_Island%2C_Bahamas%29_%2816182815352%29.jpg)

Great barracuda photo by James St. John, [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/), via [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Sphyraena_barracuda_(great_barracuda)_(Little_San_Salvador_Island,_Bahamas)_(16182815352).jpg)

## Example

Available on [PyPI](https://pypi.org/project/cudagrad/) (`pip install cudagrad`)

```py
{open('examples/example.py').read().strip()}
```

WIP! TODO: CUDA operation integration and release on PyPI

## Performance

### Tiny matmul

```
$ {cmd1}
{echo(cmd1)}
```

```
$ {cmd5}
{echo(cmd5)}
```

```
$ {cmd2}
{echo(cmd2)}
```

### Tiny backward

```
$ {cmd3}
{echo(cmd3)}
```

```
$ {cmd4}
{echo(cmd4)}
```

## License

MIT
"""

if __name__ == "__main__":
    open("README.md", "w").write(README)
