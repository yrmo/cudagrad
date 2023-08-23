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

```sql
$ sqlite3 performance.db
SQLite version 3.39.5 2022-10-14 20:58:05
Enter ".help" for usage hints.
sqlite> .headers on
sqlite> .mode column
sqlite> SELECT test.id, test.key, test.setup, test.statement, min(result.loop_nanoseconds) AS fastest_time
   ...> FROM test
   ...> INNER JOIN result ON test.id = result.id
   ...> GROUP BY test.setup, test.statement
   ...> ORDER BY test.key DESC, test.setup;
id  key            setup                   statement                                                     fastest_time
--  -------------  ----------------------  ------------------------------------------------------------  ----------------
1   tiny matmul    import cudagrad as cg;  a = cg.tensor([2, 2], [2.0, 3.0, 4.0, 5.0]); b = cg.tensor([  1.78646008399664
                                           2, 2], [6.0, 7.0, 8.0, 9.0]); c = a @ b

5   tiny matmul    import numpy as np      a = np.array([[2.0, 3.0],[4.0, 5.0]]); b = np.array([[6.0, 7  1.72451154200826
                                           .0], [8.0, 9.0]]); c = a @ b;

2   tiny matmul    import torch;           a = torch.tensor(((2.0, 3.0), (4.0, 5.0))); b = torch.tensor  5.05202025000472
                                           (((6.0, 7.0), (8.0, 9.0))); c = a @ b

3   tiny backward  import cudagrad as cg;  a = cg.tensor([2, 2], [2.0, 3.0, 4.0, 5.0]); b = cg.tensor([  2.60316520798369
                                           2, 2], [6.0, 7.0, 8.0, 9.0]); c = a @ b; d = c.sum(); d.back
                                           ward()

4   tiny backward  import torch;           a = torch.tensor(((2.0, 3.0), (4.0, 5.0)), requires_grad=Tru  22.3807172910019
                                           e); b = torch.tensor(((6.0, 7.0), (8.0, 9.0)), requires_grad
                                           =True); c = a @ b; d = c.sum(); d.backward()
```

## License

MIT
"""

if __name__ == "__main__":
    open("README.md", "w").write(README)
