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

    @unittest.skip("Not implemented")
    def test_nll_loss(self):
        t_input = torch.tensor([0.1, 0.2, 0.3, 0.4], requires_grad=True)
        t_target = torch.tensor(2)
        t_log_softmax = torch.nn.functional.log_softmax(t_input, dim=0)
        print("-" * 8)
        print(t_log_softmax)
        t_nll_loss = -t_log_softmax[t_target]
        print("-" * 8)
        print(t_nll_loss)
        t_nll_loss.backward()

        u_input = cudagrad.Tensor([4], [0.1, 0.2, 0.3, 0.4])
        # u_target = cudagrad.nn.Tensor([1], [2])
        u_target = [2]
        u_log_softmax = cudagrad.nn.log_softmax(u_input)
        print("-" * 8)
        print(u_log_softmax)
        u_nll_loss = cudagrad.nn.nll_loss(u_log_softmax, u_target)
        print("-" * 8)
        print(u_nll_loss)
        u_nll_loss.backward()

        for i in range(t_input.shape[0]):
            self.assertAlmostEqual(
                t_input.data[i].item(), u_input.data[[i]].item(), places=5
            )

        for i in range(t_input.shape[0]):
            self.assertAlmostEqual(
                t_input.grad[i].item(), u_input.grad[[i]].item(), places=5
            )

    def test_log_softmax_2(self):
        # >>> t = torch.tensor([[0.1782, 0.2920]], requires_grad=True)
        # >>> l = torch.nn.functional.log_softmax(t).sum()
        # >>> l.backward()
        # >>> t.grad
        # tensor([[ 0.0568, -0.0568]])
        # >>> l
        # tensor(-1.3895, grad_fn=<SumBackward0>)
        t = cudagrad.Tensor([1, 2], [0.1782, 0.2920])
        l = cudagrad.nn.log_softmax(t).sum()
        l.backward()
        self.assertAlmostEqual(l.item(), -1.3895, places=3)
        self.assertAlmostEqual(t.grad[[0, 0]].item(), 0.0568, places=3)
        self.assertAlmostEqual(t.grad[[0, 1]].item(), -0.0568, places=3)

    @unittest.skip("Not implemented")
    def test_nll_loss_2(self):
        # >>> t = torch.tensor([[0.1782, 0.2920]], requires_grad=True)
        # >>> target = torch.tensor([0])
        # >>>
        # >>> l = torch.nn.functional.nll_loss(t, target).sum()
        # >>> l.backward()
        # >>> t
        # tensor([[0.1782, 0.2920]], requires_grad=True)
        # >>> t.grad
        # tensor([[-1.,  0.]])
        # >>> l
        # tensor(-0.1782, grad_fn=<SumBackward0>)
        t = cudagrad.Tensor([1, 2], [0.1782, 0.2920])
        l = cudagrad.nn.nll_loss(t, [0])
        l = l.sum()
        l.backward()
        self.assertAlmostEqual(l.item(), -0.1782, places=5)
        self.assertAlmostEqual(t.grad[[0, 0]].item(), -1.0, places=5)
        self.assertAlmostEqual(t.grad[[0, 1]].item(), 0.0, places=5)

    @unittest.skip("Not implemented")
    def test_cross_entropy_loss(self):
        t = cudagrad.Tensor([1, 2], [0.1782, 0.2920])
        x = cudagrad.nn.cross_entropy(t, [0])
        x = x.sum()
        x.backward()
        self.assertAlmostEqual(x.item(), 0.7517, places=3)
        self.assertAlmostEqual(t.grad[[0, 0]].item(), -0.5284, places=3)
        self.assertAlmostEqual(t.grad[[0, 1]].item(), 0.5284, places=3)

    @unittest.skip("Not implemented")
    def test_cross_entropy_loss_mnist(self):
        """
        >>> import torch
        >>> inputs = torch.tensor([[2.0991, -0.3244, -1.4904, -0.9129, -0.1676,  0.9251,  0.1822, -0.0762,0.3743, -0.6091]], requires_grad=True)
        >>> target = torch.tensor([5])
        >>> inputs_shape = [1, 10]
        >>> target_shape = [1]
        >>> assert inputs.shape == torch.Size(inputs_shape), f"Expected inputs shape {inputs_shape}, got {inputs.shape}"
        >>> assert target.shape == torch.Size(target_shape), f"Expected target shape {target_shape}, got {target.shape}"
        >>> criterion = torch.nn.CrossEntropyLoss()
        >>> loss = criterion(inputs, target)
        >>> loss.backward()
        >>> loss
        tensor(1.9081, grad_fn=<NllLossBackward0>)
        >>> loss.grad
        <stdin>:1: UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). If you indeed want the .grad field to be populated for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. See github.com/pytorch/pytorch/pull/30531 for more informations. (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\builder\windows\pytorch\build\aten\src\ATen/core/TensorBody.h:494.)
        >>> inputs
        tensor([[ 2.0991, -0.3244, -1.4904, -0.9129, -0.1676,  0.9251,  0.1822, -0.0762,
                0.3743, -0.6091]], requires_grad=True)
        >>> inputs.grad
        tensor([[ 0.4799,  0.0425,  0.0133,  0.0236,  0.0497, -0.8516,  0.0706,  0.0545,
                0.0855,  0.0320]])
        """
        t = cudagrad.Tensor([1, 10], [2.0991, -0.3244, -1.4904, -0.9129, -0.1676,  0.9251,  0.1822, -0.0762,0.3743, -0.6091])
        x = cudagrad.nn.cross_entropy(t, [0, 5])
        x = x.sum()
        x.backward()
        self.assertAlmostEqual(x.data[[0, 0]].item(), 1.9081392288208008, places=3)
        self.assertAlmostEqual(t.grad[[0, 0]].item(), 0.4799, places=3)
        self.assertAlmostEqual(t.grad[[0, 1]].item(), 0.0425, places=3)
        self.assertAlmostEqual(t.grad[[0, 2]].item(), 0.0133, places=3)
        self.assertAlmostEqual(t.grad[[0, 3]].item(), 0.0236, places=3)
        self.assertAlmostEqual(t.grad[[0, 4]].item(), 0.0497, places=3)
        self.assertAlmostEqual(t.grad[[0, 5]].item(), -0.8516, places=3)
        self.assertAlmostEqual(t.grad[[0, 6]].item(), 0.0706, places=3)
        self.assertAlmostEqual(t.grad[[0, 7]].item(), 0.0545, places=3)
        self.assertAlmostEqual(t.grad[[0, 8]].item(), 0.0855, places=3)
        self.assertAlmostEqual(t.grad[[0, 9]].item(), 0.0320, places=3)

    @unittest.skip("Not implemented")
    def test_cross_entropy_loss_cross_backward(self):
        """
        t = torch.tensor([[-0.1600,  0.4920,  0.9304, -1.5375, -0.0106,  1.3549, -0.9282,  1.3031, -0.5426, -0.9015]], requi
        res_grad=True)
        t
        tensor([[-0.1600,  0.4920,  0.9304, -1.5375, -0.0106,  1.3549, -0.9282,  1.3031,
                -0.5426, -0.9015]], requires_grad=True)
        torch.nn.functional.cross_entropy
        <function cross_entropy at 0x000001F0CD8C1C60>
        torch.nn.functional.cross_entropy(t, torch.tensor([5]))
        tensor(1.3642, grad_fn=<NllLossBackward0>)
        l = torch.nn.functional.cross_entropy(t, torch.tensor([5]))
        l.backward()
        t
        tensor([[-0.1600,  0.4920,  0.9304, -1.5375, -0.0106,  1.3549, -0.9282,  1.3031,
                -0.5426, -0.9015]], requires_grad=True)
        t.grad
        tensor([[ 0.0562,  0.1078,  0.1672,  0.0142,  0.0652, -0.7444,  0.0261,  0.2427,
                0.0383,  0.0268]])
        l
        tensor(1.3642, grad_fn=<NllLossBackward0>)
        """
        t = cudagrad.Tensor([1, 10], [-0.1600,  0.4920,  0.9304, -1.5375, -0.0106,  1.3549, -0.9282,  1.3031, -0.5426, -0.9015])
        l = cudagrad.nn.cross_entropy(t, [0, 5])
        l.backward()
        self.assertAlmostEqual(l.item(), 1.364, places=3)
        self.assertAlmostEqual(t.grad[[0, 0]].item(), 0.0562, places=3)
        self.assertAlmostEqual(t.grad[[0, 1]].item(), 0.1078, places=3)
        self.assertAlmostEqual(t.grad[[0, 2]].item(), 0.1672, places=3)
        self.assertAlmostEqual(t.grad[[0, 3]].item(), 0.0142, places=3)
        self.assertAlmostEqual(t.grad[[0, 4]].item(), 0.0652, places=3)
        self.assertAlmostEqual(t.grad[[0, 5]].item(), -0.7444, places=3)
        self.assertAlmostEqual(t.grad[[0, 6]].item(), 0.0261, places=3)
        self.assertAlmostEqual(t.grad[[0, 7]].item(), 0.2427, places=3)
        self.assertAlmostEqual(t.grad[[0, 8]].item(), 0.0383, places=3)
        self.assertAlmostEqual(t.grad[[0, 9]].item(), 0.0268, places=3)

if __name__ == "__main__":
    unittest.main(verbosity=2)
