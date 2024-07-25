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


if __name__ == "__main__":
    unittest.main()
