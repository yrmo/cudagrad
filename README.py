import sqlite3


def table(tuples_list, headers):
    output = ""
    col_widths = [max(len(str(x)) for x in col) for col in zip(*tuples_list)]
    col_widths = [max(col_widths[i], len(headers[i])) for i in range(len(col_widths))]
    header_str = ""
    for i, header in enumerate(headers):
        header_str += header.ljust(col_widths[i] + 2)
    output += header_str + "\n"
    output += ("-" * sum(col_widths) + "-" * len(col_widths) * 2) + "\n"
    for row in tuples_list:
        row_str = ""
        for i, cell in enumerate(row):
            row_str += str(cell).ljust(col_widths[i] + 2)
        output += row_str + "\n"
    return output


connection = sqlite3.connect("performance.db")
cursor = connection.cursor()
PERFORMANCE = """
SELECT test.key, test.setup, MIN(result.loop_seconds) AS fastest_time -- , test.statement
FROM test
   INNER JOIN result
      ON test.id = result.id
GROUP BY test.setup, test.statement
ORDER BY test.key DESC, test.setup
""".strip()
cursor.execute(PERFORMANCE)
results = cursor.fetchall()
table_results = table(results, ["key", "setup", "fastest_time"])  # , 'statement'])
cursor.close()
connection.close()

README = f"""\
# cudagrad

A small tensor-valued autograd engine, inspired by [PyTorch](https://github.com/pytorch/pytorch) and [micrograd](https://github.com/karpathy/micrograd)

![](https://upload.wikimedia.org/wikipedia/commons/4/48/Sphyraena_barracuda_%28great_barracuda%29_%28Little_San_Salvador_Island%2C_Bahamas%29_%2816182815352%29.jpg)

Great barracuda photo by James St. John, [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/), via [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Sphyraena_barracuda_(great_barracuda)_(Little_San_Salvador_Island,_Bahamas)_(16182815352).jpg)

## Example

Available on [PyPI](https://pypi.org/project/cudagrad/) (`pip install cudagrad`), this requires the `nvcc` compiler!

```py
{open('examples/example.py').read().strip()}
```

WIP! TODO: CUDA operation integration and release on PyPI

## Performance

```
{table_results}
```

## License

MIT
"""

if __name__ == "__main__":
    open("README.md", "w").write(README)
