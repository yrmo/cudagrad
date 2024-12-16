import unittest

import torch

import cudagrad


class TestNN(unittest.TestCase):
    def test_softmax(self):
        t = torch.tensor([0.0, 1.0, 2.0, 4.0])
        t_softmax = torch.nn.functional.softmax(t, dim=0)

        u = cudagrad.Tensor([4], [0.0, 1.0, 2.0, 4.0])
        u_softmax = u.softmax()

        for i in range(t.shape[0]):
            self.assertAlmostEqual(
                t_softmax.data[i].item(), u_softmax.data[[i]].item(), places=5
            )

    def test_softmax_backward(self):
        t = torch.tensor([0.0, 1.0, 2.0, 4.0], requires_grad=True)
        t_softmax = torch.nn.functional.softmax(t, dim=0)
        t_softmax.sum().backward()

        u = cudagrad.Tensor([4], [0.0, 1.0, 2.0, 4.0])
        u_softmax = u.softmax().sum()
        u_softmax.backward()

        for i in range(t.shape[0]):
            self.assertAlmostEqual(
                t.grad[i].item(), u.grad[[i]].item(), places=5
            )

    def test_softmax_big(self):
        data = [0.0, 1000, 2000, 4000]

        t = torch.tensor(data)
        t_softmax = torch.nn.functional.softmax(t, dim=0)

        u = cudagrad.Tensor([4], data)
        u_softmax = u.softmax()

        for i in range(t.shape[0]):
            self.assertAlmostEqual(
                t_softmax.data[i].item(), u_softmax.data[[i]].item(), places=5
            )

    def test_cross_entropy_loss(self):
        x = cudagrad.nn.cross_entropy(
            cudagrad.Tensor([1, 2], [0.1782, 0.2920]), cudagrad.Tensor([1], [0])
        ).item()
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

    def test_cross_entropy_loss_cross_backward(self):
        """
        arr.unsqueeze(0)
        tensor([[-0.1600,  0.4920,  0.9304, -1.5375, -0.0106,  1.3549, -0.9282,  1.3031,
                -0.5426, -0.9015]])
        output = torch.tensor(arr.unsqueeze(0), requires_grad=True)
        <stdin>:1: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
        output = torch.tensor(arr.unsqueeze(0).data, requires_grad=True)
        target = torch.tensor([5], dtype=torch.long)
        loss = F.cross_entropy(output, target)
        loss
        tensor(1.3642, grad_fn=<NllLossBackward0>)
        loss.backward()
        output
        tensor([[-0.1600,  0.4920,  0.9304, -1.5375, -0.0106,  1.3549, -0.9282,  1.3031,
                -0.5426, -0.9015]], requires_grad=True)
        output.grad
        tensor([[ 0.0562,  0.1078,  0.1672,  0.0142,  0.0652, -0.7444,  0.0261,  0.2427,
                0.0383,  0.0268]])
        target
        tensor([5])
        target.grad
        """
        output = cudagrad.Tensor([1, 10], [-0.1600,  0.4920,  0.9304, -1.5375, -0.0106,  1.3549, -0.9282,  1.3031, -0.5426, -0.9015])
        target = cudagrad.Tensor([1], [5.0])
        loss = cudagrad.nn.cross_entropy(output, target)
        self.assertAlmostEqual(loss.item(), 1.364, places=3)
        loss.backward()

        self.assertAlmostEqual(target.grad[0, 0].item(), 0.0562, places=3)
        self.assertAlmostEqual(target.grad[0, 1].item(), 0.1078, places=3)
        self.assertAlmostEqual(target.grad[0, 2].item(), 0.1672, places=3)
        self.assertAlmostEqual(target.grad[0, 3].item(), 0.0142, places=3)
        self.assertAlmostEqual(target.grad[0, 4].item(), 0.0652, places=3)

        self.assertAlmostEqual(target.grad[0, 5].item(), -0.7444, places=3)
        self.assertAlmostEqual(target.grad[0, 6].item(), 0.0261, places=3)
        self.assertAlmostEqual(target.grad[0, 7].item(), 0.2427, places=3)
        self.assertAlmostEqual(target.grad[0, 8].item(), 0.0383, places=3)
        self.assertAlmostEqual(target.grad[0, 9].item(), 0.0268, places=3)

if __name__ == "__main__":
    unittest.main()
