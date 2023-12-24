from itertools import product

from cudagrad.tensor import Tensor


def mse(predicted: Tensor, actual: Tensor) -> Tensor:
    return (predicted - actual) * (predicted - actual)


class Module:
    def parameters(self):
        return [
            getattr(self, attr)
            for attr in dir(self)
            if type(getattr(self, attr)) == Tensor
        ]

    def zero_grad(self):
        for parameter in self.parameters():
            assert type(parameter) == Tensor
            parameter.zero_grad()


def sgd(model: Module, lr: float) -> None:
    def positions(tensor):
        indices = [list(range(size)) for size in tensor.size]
        for index in product(*indices):
            yield index

    for parameter in model.parameters():
        for position in positions(parameter):
            parameter.data[list(position)] = parameter.data[list(position)].item() + (
                -lr * parameter.grad[list(position)].item()
            )
