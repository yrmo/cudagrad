README = f"""\
# cudagrad

A small autograd engine

![](https://upload.wikimedia.org/wikipedia/commons/thumb/4/48/Sphyraena_barracuda_%28great_barracuda%29_%28Little_San_Salvador_Island%2C_Bahamas%29_%2816182815352%29.jpg/320px-Sphyraena_barracuda_%28great_barracuda%29_%28Little_San_Salvador_Island%2C_Bahamas%29_%2816182815352%29.jpg)

## Example

Available on [PyPI](https://pypi.org/project/cudagrad/) (`pip install cudagrad`)

```py
{open('examples/example.py').read().strip()}
```

WIP! TODO: CUDA operation integration and release on PyPI
"""

if __name__ == "__main__":
    open("README.md", "w").write(README)
