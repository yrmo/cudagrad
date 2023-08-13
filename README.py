README = f"""\
# cudagrad

A small autograd engine

WIP! TODO: CUDA operation integration and release on PyPI

## Example

Available on [PyPI](https://pypi.org/project/cudagrad/) (`pip install cudagrad`)

```py
{open('examples/example.py').read().strip()}
```
"""

if __name__ == "__main__":
    open("README.md", "w").write(README)
