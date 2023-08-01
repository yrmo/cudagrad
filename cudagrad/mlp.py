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

# %matplotlib inline

random.seed(1337)

from micrograd.engine import Value
from micrograd.nn import MLP, Layer, Neuron

# %%
if __name__ == "__main__":
    # %%
    n = MLP(3, [1, 1])
    m = MLP(3, [2, 1])
    n, m

    # %%
    class _:  # MLP
        from typing import List

        def __init__(self, nin: int, nouts: List[int]):
            sz = [nin] + nouts
            self.layers = [
                Layer(sz[i], sz[i + 1], nonlin=i != len(nouts) - 1)
                for i in range(len(nouts))
            ]

    # MLP CONSTRUCTOR
    # this is straightforward right, we make layers of
    # cin, cout combos but we only make #cout of them
    # each layers takes nin starting with nin, but then
    # all other layers take the nouts[i] of the previous

    # LINEAR OUTPUT
    # the last layers is linear so we can predict any number not just [0,int)

    # RELU LAYERS
    # ReLU (Rectified Linear Unit) introduces non-linearity into neural networks
    # due to its non-linear nature. The ReLU function is defined as:
    #
    # f(x) = max(0, x)
    #
    # without relus (or any non linear activation) nn's would be linear funtions

    #  By definition, the ReLU is 𝑚𝑎𝑥(0,𝑥). Therefore, if we split the domain from
    # (−∞,0] or [0,∞), then the function is linear. However, it's easy to see
    # that 𝑓(−1)+𝑓(1)≠𝑓(0). Hence, by definition, ReLU is not linear.
    # https://datascience.stackexchange.com/a/26481
    # %%
    def internals(n: MLP) -> None:
        for layer in n.layers:
            print(layer, "---")
            for neuron in layer.neurons:
                print(neuron, "*")
                for value in neuron.parameters():  # .w and [.b]
                    print(value, ".")

    # %%

    internals(n)
    # MLP Diagram:
    #
    #   Input (3 features)
    #       ↓
    # Layer 1 (ReLUNeuron) # w: [0.2, 0.1, 0.3], b: 0.0
    #       ↓
    # Layer 2 (LinearNeuron) # w: [0.2], b: 0.0
    #       ↓
    #  Output (1 output)

    # %%
    internals(m)
    # here's why this makes sense
    # you take three inputs, every neuron needs 3 weights plus a bias
    # but the number outputs of the first layer is two
    # so you need two neurons in the first layer
    # the linear layer gives two inputs and thus has two weights
    # %%

    xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
    ys = [1.0, -1.0, -1.0, 1.0]
    # %%

    nn = MLP(3, [4, 4, 1])  # i think this ignores our random seed sadly
    for _ in range(50):
        ypred = [nn(x) for x in xs]
        loss = sum((a - b) ** 2 for a, b in zip(ypred, ys))
        for p in nn.parameters():
            p.grad = 0.0
        loss.backward()
        for p in nn.parameters():
            p.data += -0.05 * p.grad
        print(_ + 1, loss.data, [f"{x.data:1.2f}" for x in ypred])
    # %%

    import random

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

        def __repr__(self):
            return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

    # %%
    print(Neuron(2)([0.5, 0.2]).data)
    print(Neuron(2)([tensor([1], [0.5]), tensor([1], [0.2])]).data)
    # Layer(1, 2)
    # MLP(2, [2, 1])

    # # %%
    # class Layer(Module):
    #     def __init__(self, nin, nout, **kwargs):
    #         self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    #     def __call__(self, x):
    #         out = [n(x) for n in self.neurons]
    #         return out[0] if len(out) == 1 else out

    #     def parameters(self):
    #         return [p for n in self.neurons for p in n.parameters()]

    #     def __repr__(self):
    #         return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

    # class MLP(Module):
    #     def __init__(self, nin, nouts):
    #         sz = [nin] + nouts
    #         self.layers = [
    #             Layer(sz[i], sz[i + 1], nonlin=i != len(nouts) - 1)
    #             for i in range(len(nouts))
    #         ]

    #     def __call__(self, x):
    #         for layer in self.layers:
    #             x = layer(x)
    #         return x

    #     def parameters(self):
    #         return [p for layer in self.layers for p in layer.parameters()]

    #     def __repr__(self):
    #         return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

    # # %%
    # # nn = MLP(3, [4, 4, 1])  # i think this ignores our random seed sadly
    # # for _ in range(50):
    # #     ypred = [nn(x) for x in xs]
    # #     loss = sum((a - b) ** 2 for a, b in zip(ypred, ys))
    # #     for p in nn.parameters():
    # #         p.grad = 0.0
    # #     loss.backward()
    # #     for p in nn.parameters():
    # #         p.data += -0.05 * p.grad
    # #     print(_ + 1, loss.data, [f"{x.data:1.2f}" for x in ypred])
