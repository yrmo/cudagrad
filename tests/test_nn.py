import unittest

import torch
import torch.nn as nn

import cudagrad


class TestNN(unittest.TestCase):
    def test_softmax(self):
        t = torch.tensor([0.0, 1.0, 2.0, 4.0])
        t_softmax = torch.nn.functional.softmax(t, dim=0)

        u = cudagrad.Tensor([4], [0.0, 1.0, 2.0, 4.0])
        u_softmax = cudagrad.nn.softmax(u)

        for i in range(t.shape[0]):
            self.assertAlmostEqual(
                t_softmax.data[i].item(), u_softmax.data[[i]].item(), places=5
            )

    def test_softmax_big(self):
        data = [0.0, 1000, 2000, 4000]

        t = torch.tensor(data)
        t_softmax = torch.nn.functional.softmax(t, dim=0)

        u = cudagrad.Tensor([4], data)
        u_softmax = cudagrad.nn.softmax(u)

        for i in range(t.shape[0]):
            self.assertAlmostEqual(
                t_softmax.data[i].item(), u_softmax.data[[i]].item(), places=5
            )

    def test_cross_entropy_loss(self):
        logits = torch.tensor([[0.7, 1.3], [1.1, 0.9], [0.2, 2.0]])
        targets = torch.tensor([1, 0, 1])
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, targets)
        self.assertAlmostEqual(loss.item(), 42)

if __name__ == "__main__":
    unittest.main()
