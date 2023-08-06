# cudagrad

A small autograd engine

Work In Progress! TODO: CUDA operation integration and release on PyPI

## Example

```cpp
// c++ -std=c++11 -I./src examples/example.cpp && ./a.out
#include "tensor.hpp"
int main() {
  auto a = cg::tensor({2, 2}, {2.0, 3.0, 4.0, 5.0});
  auto b = cg::tensor({2, 2}, {6.0, 7.0, 8.0, 9.0});
  auto c = cg::tensor({2, 2}, {10.0, 10.0, 10.0, 10.0});
  auto d = cg::tensor({2, 2}, {11.0, 11.0, 11.0, 11.0});
  auto e = (a.get()->matmul(b) + c) * d;
  auto f = e.get()->sum();
  f.get()->backward();

  using namespace std;
  for (auto& x : f.get()->data_) {cout<<x<<" ";} // 2794
  for (auto& x : f.get()->size_) {cout<<x<<" ";} // 1
  for (auto& x : a.get()->grad_) {cout<<x<<" ";} // 143 187 143 187
  for (auto& x : b.get()->grad_) {cout<<x<<" ";} // 66 66 88 88
}
```

Available on [PyPI](https://pypi.org/project/cudagrad/), use `pip install cudagrad` to get the Python bindings

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

## Design

~~The plan is to be similar to PyTorch's internals, particularily the [Variable/Tensor Merge Proposal](https://github.com/pytorch/pytorch/issues/13638) design.~~ The design is a mix of PyTorch and micrograd, with micrograd like members and PyTorch like backward classes with an `void apply(std::shared_ptr<Tensor> grad_output, std::vector<std::shared_ptr<Tensor>> grad_inputs)` interface.

For simplicity, many features PyTorch has cudagrad does not, like broadcasting and views. All operations are defined only on `std::shared_ptr<Tensor>`, for now at least.

### Tensor

The `/cudagrad/src` folder contains the `Tensor` class written in C++ and CUDA, and it's pybind11 bindings. If you install the `cudagrad` package locally (using `python -m pip install -e .`) you will see a tensor `.so` file in the `/cudagrad/cudagrad` Python package folder, this is a normal Python package called `tensor` that you can import:

```py
$ pwd
/Users/ryan/cudagrad/cudagrad
$ ls
__init__.py			mlp.py
__pycache__			tensor.cpython-310-darwin.so
$ python -q
>>> from tensor import tensor
>>> tensor
<built-in method tensor of PyCapsule object at 0x100f81c80>
>>> tensor([1], [42.0])
tensor([1], [42])
```

It's contains the `Tensor` class, and the `tensor` factory function. In the C++, the `tensor` factory is necessary because the constructor of a type `T` cannot have it's own constructor create `std::shared_ptr<T>`, but in Python `tensor` and `Tensor` do the same thing:

```py
>>> from tensor import tensor, Tensor
>>> tensor
<built-in method tensor of PyCapsule object at 0x1015ca1f0>
>>> Tensor
<class 'tensor.Tensor'>
>>> tensor([1], [42])
tensor([1], [42])
>>> Tensor([1], [42])
tensor([1], [42])
```

### Neural networks

To improve the `Tensor` class and it's Python bindings `cudagrad` tries to eat it's own dog food and implement the

#### Multi-Layer Perceptron

An MLP implementation can be found at `/cudagrad/cudagrad/mlp.py`.

TODO example

#### Transformer

TODO

## Goals

The goal of this project is to learn more about PyTorch's internals, neural networks, and C++. And some CUDA too!

To do this, I'll gradually add support to cudagrad for the mathematical operations required to create the expression graph of various neural networks. The long term goals are to implement a Multilayer perceptron by the summer of 2023, and a Transformer by end of the year.

> "Maybe it's a bad idea to have really big ambitions initially. Because the bigger your ambitions, the longer they're going to take to realize, and the longer you're projecting into the future, the more likely you're going to be wrong."
>
> [paulg @ PyCon US 2012](https://youtu.be/R9ITLdmfdLI?t=1927)

## Setup

- TODO: CUDA driver setup
- TODO: CUDA toolkit setup
- TODO: Check all system requirements met (nvcc, git, cmake, make...)

## Running tests

Taking inspiration from [micrograd's tests](https://github.com/karpathy/micrograd/blob/master/test/test_engine.py), we will use [PyTorch's C++ frontend](https://pytorch.org/cppdocs/frontend.html) for high level sanity checks using GoogleTest.

To run the tests use:

```sh
python makefile.py test
```

Running the tests requires: `cmake`, `make`, `torch` installed (on the version of Python accessed by the `python` command), `git`, and a C++ compiler. Note that these requirements are only for when you need to run the tests, otherwise except the C++ compiler they are not needed.
