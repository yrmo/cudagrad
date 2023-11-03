# cudagrad

A small tensor-valued autograd engine, inspired by [PyTorch](https://github.com/pytorch/pytorch) and [micrograd](https://github.com/karpathy/micrograd)

![](https://upload.wikimedia.org/wikipedia/commons/4/48/Sphyraena_barracuda_%28great_barracuda%29_%28Little_San_Salvador_Island%2C_Bahamas%29_%2816182815352%29.jpg)

Great barracuda photo by James St. John, [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/), via [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Sphyraena_barracuda_(great_barracuda)_(Little_San_Salvador_Island,_Bahamas)_(16182815352).jpg)

## Example

Available on [PyPI](https://pypi.org/project/cudagrad/) (`pip install cudagrad`)

```py
# python -m pip install cudagrad; python ./examples/example.py
import cudagrad as cg

a = cg.Tensor([2, 2], [2.0, 3.0, 4.0, 5.0])
b = cg.Tensor([2, 2], [6.0, 7.0, 8.0, 9.0])
c = cg.Tensor([2, 2], [10.0, 10.0, 10.0, 10.0])
d = cg.Tensor([2, 2], [11.0, 11.0, 11.0, 11.0])
e = ((a @ b) + c) * d
f = e.sum()
f.backward()

print(f.data)  # [2794.0]
print(f.size)  # [1]
print(a.grad)  # [143.0, 187.0, 143.0, 187.0]
print(b.grad)  # [66.0, 66.0, 88.0, 88.0]
```

WIP! TODO: CUDA operation integration and release on PyPI

## Performance

```
key            setup                   fastest_time            
---------------------------------------------------------------
tiny matmul    import cudagrad as cg;  1.6817973330034873e-06  
tiny matmul    import numpy as np      1.7551545829628593e-06  
tiny matmul    import torch;           5.315409083035774e-06   
tiny backward  import cudagrad as cg;  2.59421900002053e-06    
tiny backward  import torch;           2.302449666702887e-05   
big matmul     import cudagrad as cg;  1.215179824992083       
big matmul     import torch;           0.003011275001335889    

```

## License

MIT
