import torch
import torch.nn as nn


class SingleNeuron(nn.Module):
    def __init__(self):
        super(SingleNeuron, self).__init__()
        self.weight = nn.Parameter(torch.randn(1))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        out = self.weight * x + self.bias
        # out = torch.sigmoid(out)
        return out


neuron = SingleNeuron()
print(neuron.state_dict())
