from random import random

from cudagrad.tensor import Tensor # type: ignore



class Linear:
    def __init__(self, inputs: int, outputs: int):
        self.w = Tensor([outputs, inputs], [random() for _ in range(outputs * inputs)])
        self.b = Tensor([outputs], [random() for _ in range(outputs)])

    def __call__(self, x: Tensor) -> Tensor:
        return (self.w @ x)+ self.b

if __name__ == "__main__":
    # OR
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    targets = [0, 1, 1, 1]
    
    model = Linear(2, 2)
    print(model(Tensor([2, 1], inputs[0])))