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

from cudagrad.tensor import Tensor


def mse(predicted: Tensor, actual: Tensor) -> Tensor:
    # TODO this assumes column tensor
    difference = predicted - actual
    squared = difference * difference
    summed = squared.sum()
    mean = summed / Tensor([1], [summed.size[0]])
    return mean


class Module:
    def parameters(self) -> list[Tensor]:
        params = []
        for x in self.__dict__:
            param = getattr(self, x)
            if type(param) is Tensor:
                params.append(param)
        return params

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            assert type(parameter) is Tensor
            parameter.zero_grad()


def sgd(model: Module, lr: float) -> None:
    for parameter in model.parameters():
        parameter.data[:] = parameter + (
            Tensor.explode(parameter.size, -lr) * parameter.grad()
        )

def log_softmax(t: Tensor) -> Tensor:
    return Tensor.log(t.softmax())

def nll_loss(inputs: Tensor, target: list[int]) -> Tensor:
    print(1)
    # inputs is [num_classes] and log-softmaxed
    selected = inputs.select(target)
    print(2)
    return Tensor([1], [-1.0]) * selected

def cross_entropy(inputs: Tensor, target: Tensor) -> Tensor:
    log_softmaxed = log_softmax(inputs)
    return nll_loss(log_softmaxed, target)
