README = f"""\
# cudagrad

A small autograd engine

![](parallel_1_20.png)

## Example

```cpp
{open('examples/example.cpp').read().strip()}
```

Available on [PyPI](https://pypi.org/project/cudagrad/), use `pip install cudagrad` to get the Python bindings

```py
{open('examples/example.py').read().strip()}
```

## Design

~~The plan is to be similar to PyTorch's internals, particularily the [Variable/Tensor Merge Proposal](https://github.com/pytorch/pytorch/issues/13638) design.~~ The design is a mix of PyTorch and micrograd, with micrograd like members and PyTorch like backward classes with an `apply()` interface.

For simplicity, many features PyTorch has cudagrad does not, like broadcasting and views. All operations are defined only on `std::shared_ptr<Tensor>`, for now at least.

## Goals

The goal of this project is to learn more about PyTorch's internals, neural networks, and C++. And some CUDA too!

To do this, I'll gradually add support to cudagrad for the mathematical operations required to create the expression graph of various neural networks. The long term goals are to implement a Multilayer perceptron by the summer of 2023, and a Transformer by end of the year.

> "Maybe it's a bad idea to have really big ambitions initially. Because the bigger your ambitions, the longer they're going to take to realize, and the longer you're projecting into the future, the more likely you're going to be wrong."
>
> [paulg @ PyCon US 2012](https://youtu.be/R9ITLdmfdLI?t=1927)

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
