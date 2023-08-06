README = f"""\
# cudagrad

A small autograd engine

Work In Progress! TODO: CUDA operation integration and release on PyPI

## Example

```cpp
{open('examples/example.cpp').read().strip()}
```

Available on [PyPI](https://pypi.org/project/cudagrad/), use `pip install cudagrad` to get the Python bindings

```py
{open('examples/example.py').read().strip()}
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

In C++, constructors directly return an instance of the class type. However, if we want to manage the lifetime of an object using `std::shared_ptr<T>`, we typically use a factory function. In this case, the `tensor` factory function is used for creating `std::shared_ptr<Tensor>` instances in C++, but in Python, the distinction is abstracted away by the pybind11 bindings, allowing both `tensor` and `Tensor` to do the same thing.

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
"""

if __name__ == "__main__":
    open("README.md", "w").write(README)
