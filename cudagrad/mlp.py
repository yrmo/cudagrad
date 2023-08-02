# type: ignore
#
# Copyright 2023 Ryan Moore
#
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

# %%
import random
from typing import *  # type: ignore

from .cudagrad_bindings import *  # type: ignore

# wait, can cudagrad use python's global seed randomness?
# probably hard/impossible to comunicate back to c++
# horrible, good idea!
random.seed(1337)



# import cudagrad
# from cudagrad import Tensor, tensor


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, nin, nonlin=True):
        self.w = [tensor([1], [random.uniform(-1, 1)]) for _ in range(nin)]
        self.b = tensor([1], [0])
        self.nonlin = nonlin

    def __call__(self, x):
        ans = tensor([1], [0])
        for elem_x in x:
            for elem_w in self.w:
                if type(elem_x) != Tensor:
                    elem_x = tensor([1], [elem_x])
                ans = ans + (elem_w * elem_x)
        ans = ans + self.b
        return ans
        # act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        # return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def zero_grad(self):
        for tensor in self.w + [self.b]:
            tensor.zero_grad()

    def train(self, rate: float):
        for tensor in self.w + [self.b]:
            assert tensor.size == [1]
            # FIXME this is a must fix, I must be able to assign to data!
            # >>> cg
            # <module 'cudagrad' from '/Users/ryan/cudagrad/cudagrad/__init__.py'>
            # >>> x = cg.tensor([1], [42])
            # >>> x
            # [42]
            # [0]
            # >>> x.data
            # [42.0]
            # >>> x.data = [1]
            # Traceback (most recent call last):
            # File "<stdin>", line 1, in <module>
            # AttributeError: can't set attribute
            # >>> x.data[0]
            # 42.0
            # >>> x.data[0] = 1
            # >>> x
            # [42]
            # [0]
            before = tensor.data[0]
            update_value = tensor.data[0] + rate * tensor.grad[0]
            tensor.data[0] = tensor.data[0] + rate * tensor.grad[0]
            print(before, '->', tensor.data[0])
            # print(tensor.data[0])
            # print('after', tensor.data[0])

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


# %%
print(Neuron(2)([0.5, 0.2]).data)
print(Neuron(2)([tensor([1], [0.5]), tensor([1], [0.2])]).data)
# Layer(1, 2)
# MLP(2, [2, 1])

# %%
class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def zero_grad(self):
        for neuron in self.neurons:
            neuron.zero_grad()

    def train(self, rate: float):
        for neuron in self.neurons:
            neuron.train(rate)

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


l = Layer(3, 2)

class MLP(Module):
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [
            Layer(sz[i], sz[i + 1], nonlin=i != len(nouts) - 1)
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()

    def train(self, rate: float):
        for layer in self.layers:
            layer.train(rate)

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

# %%

# TODO make commented out part work make it a test

xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
ys = [1.0, -1.0, -1.0, 1.0]
# %%

# TODO make Mean Squared Error loss function some python function
#      redoing this is annoying how does pytorch store loss functions?

nn = MLP(3, [4, 4, 1])  # i think this ignores our random seed sadly
for _ in range(50):

    ypred = [nn(x) for x in xs]
    # print(ypred, ys)
    # print(list(zip(ypred, ys)))

    # TODO ** POW would be nice here
    # TODO no idea why sum() doesnt work
    tensor_ys = [tensor([1], [float(y)]) for y in ys]
    loss = tensor([1], [0.0])
    for x in [(a - b) * (a - b) for a, b in zip(ypred, tensor_ys)]:
        loss = loss + x
    # loss = sum((a - b) * (a - b) for a, b in zip(ypred, tensor_ys))

    nn.zero_grad()
    # for p in nn.parameters():
    #     p.grad = 0.0

    loss.backward()
    nn.train(-0.05)
    # for p in nn.parameters():
    #     p.data += -0.05 * p.grad

    # print(_ + 1, loss.data[0], [f"{y.data[0]:1.2f}" for y in ypred])
