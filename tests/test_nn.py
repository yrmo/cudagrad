import unittest

import torch

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
        x = cudagrad.nn.cross_entropy(
            cudagrad.Tensor([1, 2], [0.1782, 0.2920]), cudagrad.Tensor([1], [0])
        ).item()
        self.assertAlmostEqual(x, 0.7517, places=3)


if __name__ == "__main__":
    unittest.main()
