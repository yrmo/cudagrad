# So the magic is that there's this relatively simple algorithm called
# backpropagation that takes the error in the output and sends that
# error backwards through the network and computes through all the
# connections how you should change them to improve the behavior, and
# then you change them all a tiny bit and you just keep going with
# another example. And surprisingly that actually works. For many years
# people thought that would just get jammed up — it would get stuck
# somewhere — but no it doesn't, it actually works very well.
#
# Geoffrey Hinton

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
