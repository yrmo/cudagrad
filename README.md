# About 

[cudagrad](https://pypi.org/project/cudagrad/) is a tensor-valued autograd engine for Python

Any ideas? Open an [issue](https://github.com/yrmo/cudagrad/issues), or email me: <a href="mailto:ryanm.inbox@gmail.com">ryanm.inbox@gmail.com</a>!


# Installation

Use `pip install cudagrad`. As a warning, NVIDIA's [`nvcc`](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html) compiler must be installed on the system for `pip install cudagrad` to work, as cudagrad is a [C++ extension to Python](https://docs.python.org/3/extending/building.html) (using [pybind11](https://github.com/pybind/pybind11)).


```python
import cudagrad as cg

cg
```




    <module 'cudagrad' from '/home/ryan/.pyenv/versions/3.10.13/lib/python3.10/site-packages/cudagrad/__init__.py'>



# Tensor

cudagrad tensors are like PyTorch tensors, except:

- Tensors only use `float32`
- Tensors `requires_grad` by default
- The `Tensor` constructor takes two lists instead of a nested list: cg.Tensor([size], [data])

## Tensor `__init__`

The data list is loaded in [row-major order](https://en.wikipedia.org/wiki/Row-_and_column-major_order) (left to right, top to bottom)


```python
T = cg.Tensor([2, 1], range(2))
T
```




    <cudagrad.Tensor([2, 1, ], [0, 1, ]) object at 0x5640b56640e0>



Great! We made a tensor that is a column matrix with the values of 0, and 1. This would be the same as the following in PyTorch for example:


```python
import torch

torch.tensor([[0], [1]], dtype=torch.float32, requires_grad=True)
```




    tensor([[0.],
            [1.]], requires_grad=True)



If we `print` this tensor two matrixes are printed, first the `data`, then the `grad`:


```python
T.size
```




    [2, 1]




```python
print(T)
```

    [[0],
     [1]]
    [[0],
     [0]]


Various operations are supported, far fewer than PyTorch, but I plan to grow this over time... At the moment some basics are supported:


```python
loss = (T + T).sum()
loss
```




    <cudagrad.Tensor([1, ], [2, ]) object at 0x5640baa3c700>



You might wondering why I show the address of the Tensor object, unlike PyTorch. It's because it's helpful for debugging, I use this myself for cudagrad's development.


```python
loss.graph()
```

    0x5640baa3c700 s
      0x5640baa3c640 +
        0x5640b56640e0  
        0x5640b56640e0  


I'm a big fan of introspection.

## Tensor Methods

Below is some gross stuff to turn the `help(cg.Tensor)` into a string.


```python
import contextlib
import io
import re

with io.StringIO() as buf, contextlib.redirect_stdout(buf):
    help(cg.Tensor)
    HELP = re.split("-{5,}", buf.getvalue())
```


```python
[x[2:].strip() for x in HELP[0].splitlines() if "(self:" in x]
```




    ['__add__(self: cudagrad.tensor.Tensor, arg0: cudagrad.tensor.Tensor) -> cudagrad.tensor.Tensor',
     '__getitem__(self: cudagrad.tensor.Tensor, arg0: List[int]) -> cudagrad.tensor.Tensor',
     '__init__(self: cudagrad.tensor.Tensor, arg0: List[int], arg1: List[float]) -> None',
     '__matmul__(self: cudagrad.tensor.Tensor, arg0: cudagrad.tensor.Tensor) -> cudagrad.tensor.Tensor',
     '__mul__(self: cudagrad.tensor.Tensor, arg0: cudagrad.tensor.Tensor) -> cudagrad.tensor.Tensor',
     '__repr__(self: cudagrad.tensor.Tensor) -> str',
     '__setitem__(self: cudagrad.tensor.Tensor, arg0: List[int], arg1: float) -> None',
     '__str__(self: cudagrad.tensor.Tensor) -> str',
     '__sub__(self: cudagrad.tensor.Tensor, arg0: cudagrad.tensor.Tensor) -> cudagrad.tensor.Tensor',
     '__truediv__(self: cudagrad.tensor.Tensor, arg0: cudagrad.tensor.Tensor) -> cudagrad.tensor.Tensor',
     'backward(self: cudagrad.tensor.Tensor) -> None',
     'foo(self: int) -> int',
     'get_shared(self: cudagrad.tensor.Tensor) -> cudagrad.tensor.Tensor',
     'graph(self: cudagrad.tensor.Tensor) -> None',
     'item(self: cudagrad.tensor.Tensor) -> float',
     'relu(self: cudagrad.tensor.Tensor) -> cudagrad.tensor.Tensor',
     'sum(self: cudagrad.tensor.Tensor) -> cudagrad.tensor.Tensor',
     'zero_grad(self: cudagrad.tensor.Tensor) -> None']



Right now this includes the barebones to make a Multi-Layer perceptron:


```python
# python -m pip install cudagrad; python ./examples/example.py
import cudagrad as cg

a = cg.Tensor([2, 2], [2.0, 3.0, 4.0, 5.0])
b = cg.Tensor([2, 2], [6.0, 7.0, 8.0, 9.0])
c = cg.Tensor([2, 2], [10.0, 10.0, 10.0, 10.0])
d = cg.Tensor([2, 2], [11.0, 11.0, 11.0, 11.0])
e = cg.Tensor.relu(((a @ b) + c) * d)
f = e.sum()
f.backward()

print(f.data[[0]])  # awful I know, working on it!
print(f.size)
print(a.grad)
print(b.grad)
```

    [2794]
    [0]
    [1]
    [143.0, 187.0, 143.0, 187.0]
    [66.0, 66.0, 88.0, 88.0]



```python
import torch

at = torch.tensor(((2.0, 3.0), (4.0, 5.0)), requires_grad=True)
bt = torch.tensor(((6.0, 7.0), (8.0, 9.0)), requires_grad=True)
ct = torch.tensor(((10.0, 10.0), (10.0, 10.0)), requires_grad=True)
dt = torch.tensor(((11.0, 11.0), (11.0, 11.0)), requires_grad=True)
et = torch.relu(((at @ bt) + ct) * dt)
ft = et.sum()
ft.backward()

print(ft.data)
print(ft.size())
print(at.grad)
print(bt.grad)
```

    tensor(2794.)
    torch.Size([])
    tensor([[143., 187.],
            [143., 187.]])
    tensor([[66., 66.],
            [88., 88.]])


## Tensor static methods


```python
[x[2:].strip() for x in HELP[1].splitlines() if "(arg0:" in x]
```




    ['explode(arg0: List[int], arg1: float) -> cudagrad.tensor.Tensor',
     'ones(arg0: List[int]) -> cudagrad.tensor.Tensor',
     'rand(arg0: List[int]) -> cudagrad.tensor.Tensor',
     'zeros(arg0: List[int]) -> cudagrad.tensor.Tensor']



These turn out to be very helpful, `exlpode` is the only way to do broadcast at the moment:


```python
cg.Tensor.explode([2], 4.2)
```




    <cudagrad.Tensor([2, ], [4.2, 4.2, ]) object at 0x5640baa461f0>



## Tensor readonly properties


```python
[x[2:].strip() for x in HELP[2].splitlines()[2:] if x[2:].strip() != ""]
```




    ['data', 'grad', 'size']



This is what that would look like using PyTorch:

# Neural Networks

The neural networks this project provides will be written purely in Python, using only the `cudagrad.Tensor`. Ideally, this helps improve the `Tensor` class over time, as it is used to create increasingly complicated neural networks by eating our own dogfood!

## Multi-Layer Perceptron

Work in progress!
