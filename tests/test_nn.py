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
        x = cudagrad.nn.cross_entropy(cudagrad.Tensor([1, 2], [0.1782, 0.2920]), cudagrad.Tensor([1], [0])).item()
        self.assertAlmostEqual(x, 0.7517, places=3)


    def test_cross_entropy_loss_mnist(self):
        """
        import torch
        import torch.nn as nn
        inputs = torch.tensor([[2.0991, -0.3244, -1.4904, -0.9129, -0.1676,  0.9251,  0.1822, -0.0762,0.3743, -0.6091]])
        target = torch.tensor([5])
        inputs_shape = [1, 10]
        target_shape = [1]
        assert inputs.shape == torch.Size(inputs_shape), f"Expected inputs shape {inputs_shape}, got {inputs.shape}"
        assert target.shape == torch.Size(target_shape), f"Expected target shape {target_shape}, got {target.shape}"
        criterion = nn.CrossEntropyLoss()
        loss = criterion(inputs, target)
        print(loss.item()) # 1.9081392288208008
        """
        x = cudagrad.nn.cross_entropy(cudagrad.Tensor([1, 10], [2.0991, -0.3244, -1.4904, -0.9129, -0.1676,  0.9251,  0.1822, -0.0762,0.3743, -0.6091]), cudagrad.Tensor([1], [5])).item()
        self.assertAlmostEqual(x, 1.9081392288208008, places=3)

if __name__ == "__main__":
    unittest.main()
