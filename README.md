<div align="center">
<h1>
    <div>cudagrad</div>
</h1>

CUDA C++ strided float tensor automatic differentiation engine with Python bindings

</div>

# Install

```
pip install cudagrad
```

# Examples

The following examples were written purely in Python using only [`cudagrad.Tensor`](./Tensor.ipynb) for learning:


### OR


![](benchmarks/_cudagrad/plots/or.jpg)

0.57 seconds (55.3% slower than `torch`)

[`/benchmarks/_cudagrad/or.py`](https://github.com/yrmo/cudagrad/blob/main/benchmarks/_cudagrad/or.py)


### XOR


![](benchmarks/_cudagrad/plots/xor.jpg)

4.83 seconds (0.0% faster than `torch`)

[`/benchmarks/_cudagrad/xor.py`](https://github.com/yrmo/cudagrad/blob/main/benchmarks/_cudagrad/xor.py)


### MOONS


![](benchmarks/_cudagrad/plots/moons.jpg)

12.79 seconds (0.0% faster than `torch`)

[`/benchmarks/_cudagrad/moons.py`](https://github.com/yrmo/cudagrad/blob/main/benchmarks/_cudagrad/moons.py)


### MNIST


![](benchmarks/_cudagrad/plots/mnist.jpg)

3.52 seconds (0.0% faster than `torch`)

[`/benchmarks/_cudagrad/mnist.py`](https://github.com/yrmo/cudagrad/blob/main/benchmarks/_cudagrad/mnist.py)

