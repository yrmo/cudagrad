from os import environ, system
from pathlib import Path
from pstats import Stats
from textwrap import dedent

README = f"""\
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

"""


def profile(examples: list[str], x):
    global README
    for example in examples:
        # system(f"python -m cProfile -o ./benchmarks/{x}/profiles/{example}.prof ./benchmarks/{x}/{example}.py")
        p = Stats(f"./benchmarks/_{x}/profiles/{example}.prof")
        README = README + dedent(f"""\

### {example.upper()}

![](benchmarks/_{x}/plots/{example}.jpg)

[`/benchmarks/_{x}/{example}.py`](https://github.com/yrmo/cudagrad/blob/main/benchmarks/_cudagrad/{example}.py) (in {round(p.total_tt, 2)} seconds)

""")


if __name__ == "__main__":
    environ["PROFILING"] = "1"
    profile([x.stem for x in list(Path('.').glob('./benchmarks/_cudagrad/*.py'))], "cudagrad")
    # profile([x.stem for x in list(Path('.').glob('./examples/_torch/*.py'))], "torch")

    with open("README.md", "w") as f:
        f.write(README)
