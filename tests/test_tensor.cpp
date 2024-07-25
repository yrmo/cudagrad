// Copyright 2023-2024 Ryan Moore

#include <gtest/gtest.h>

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "../src/tensor.hpp"  // NOLINT (build/include_subdir)

TEST(Basic, Shared) {
  auto foo = cg::tensor({1}, {42.0});
  auto bar = foo->get_shared();

  EXPECT_EQ(foo, bar);
}

TEST(Basic, Equal) {
  auto foo = cg::tensor({1}, {42.0});
  auto bar = cg::tensor({1}, {24.0});
  EXPECT_EQ((foo == foo).get()->data_[0], 1.0);
  EXPECT_EQ((foo == bar).get()->data_[0], 0.0);
}

TEST(Basic, NotEqual) {
  auto foo = cg::tensor({1}, {42.0});
  auto bar = cg::tensor({1}, {24.0});
  EXPECT_EQ((foo != foo).get()->data_[0], 0.0);
  EXPECT_EQ((foo != bar).get()->data_[0], 1.0);
}

TEST(Basic, Add) {
  auto a = cg::tensor({1}, {42.0});
  auto b = cg::tensor({1}, {42.0});
  auto c = a + b;
  c.get()->backward();

  EXPECT_EQ(c.get()->data_[0], 84.0);
  EXPECT_EQ(a.get()->grad_[0], 1.0);
  EXPECT_EQ(b.get()->grad_[0], 1.0);
}

TEST(Basic, Sum) {
  /*
  >>> import torch
  >>> a = torch.tensor((42.0, 24.0, 12.0), requires_grad=True)
  >>> l = a.sum()
  >>> l.backward()
  >>> l.data
  tensor(78.)
  >>> a.grad
  tensor([1., 1., 1.])
  */
  auto a = cg::tensor({3}, {42.0, 24.0, 12.0});
  auto l = a.get()->sum();
  l.get()->backward();

  EXPECT_EQ(l.get()->data_[0], 78.0);
  EXPECT_EQ(l.get()->grad_.size(), 1);
  EXPECT_EQ(a.get()->grad_.size(), 3);
  EXPECT_EQ(a.get()->grad_[0], 1.0);
  EXPECT_EQ(a.get()->grad_[1], 1.0);
  EXPECT_EQ(a.get()->grad_[2], 1.0);
}

TEST(Basic, Max) {
  /*
  >>> import torch
  >>> a = torch.tensor([1,2,1,3,1.], requires_grad=True)
  >>> l = a.max()
  >>> l.backward()
  >>> l.data
  tensor(3.)
  >>> a.grad
  tensor([0., 0., 0., 1., 0.])
  */
  auto a = cg::tensor({5}, {1.0, 2.0, 1.0, 3.0, 1.0});
  auto l = a.get()->max();
  l.get()->backward();

  EXPECT_EQ(l.get()->data_[0], 3.0);
  EXPECT_EQ(l.get()->grad_.size(), 1);
  EXPECT_EQ(a.get()->grad_.size(), 5);
  EXPECT_EQ(a.get()->grad_[0], 0.0);
  EXPECT_EQ(a.get()->grad_[1], 0.0);
  EXPECT_EQ(a.get()->grad_[2], 0.0);
  EXPECT_EQ(a.get()->grad_[3], 1.0);
  EXPECT_EQ(a.get()->grad_[4], 0.0);
}

TEST(Basic, Minus) {
  /*
  >>> a = torch.tensor((5.0, 4.0, 3.0, 2.0), requires_grad=True)
  >>> b = torch.tensor((2.0, 3.0, 4.0, 5.0), requires_grad=True)
  >>> a
  tensor([5., 4., 3., 2.], requires_grad=True)
  >>> b
  tensor([2., 3., 4., 5.], requires_grad=True)
  >>> c = a - b
  >>> l = c.sum()
  >>> l.backward()
  >>> l
  tensor(0., grad_fn=<SumBackward0>)
  >>> a.grad
  tensor([1., 1., 1., 1.])
  >>> b.grad
  tensor([-1., -1., -1., -1.])
  */
  auto a = cg::tensor({4}, {5.0, 4.0, 3.0, 2.0});
  auto b = cg::tensor({4}, {2.0, 3.0, 4.0, 5.0});
  auto c = a - b;
  auto l = c.get()->sum();
  l.get()->backward();

  EXPECT_EQ(l.get()->data_[0], 0.0);
  EXPECT_EQ(l.get()->grad_.size(), 1);

  EXPECT_EQ(a.get()->grad_.size(), 4);
  EXPECT_EQ(b.get()->grad_.size(), 4);

  EXPECT_EQ(a.get()->grad_[0], 1.0);
  EXPECT_EQ(a.get()->grad_[1], 1.0);
  EXPECT_EQ(a.get()->grad_[2], 1.0);
  EXPECT_EQ(a.get()->grad_[3], 1.0);

  EXPECT_EQ(b.get()->grad_[0], -1.0);
  EXPECT_EQ(b.get()->grad_[1], -1.0);
  EXPECT_EQ(b.get()->grad_[2], -1.0);
  EXPECT_EQ(b.get()->grad_[3], -1.0);
}

TEST(Basic, Multiply) {
  /*
  >>> a = torch.tensor((5.0, 4.0, 3.0, 2.0), requires_grad=True)
  >>> b = torch.tensor((2.0, 3.0, 4.0, 5.0), requires_grad=True)
  >>> c = a * b
  >>> l = c.sum()
  >>> l
  tensor(44., grad_fn=<SumBackward0>)
  >>> l.backward()
  >>> l
  tensor(44., grad_fn=<SumBackward0>)
  >>> a.grad
  tensor([2., 3., 4., 5.])
  >>> b.grad
  tensor([5., 4., 3., 2.])
  */
  auto a = cg::tensor({4}, {5.0, 4.0, 3.0, 2.0});
  auto b = cg::tensor({4}, {2.0, 3.0, 4.0, 5.0});
  auto c = a * b;
  auto l = c.get()->sum();
  l.get()->backward();

  EXPECT_EQ(l.get()->data_[0], 44.0);
  EXPECT_EQ(l.get()->grad_.size(), 1);

  EXPECT_EQ(a.get()->grad_.size(), 4);
  EXPECT_EQ(b.get()->grad_.size(), 4);

  EXPECT_EQ(a.get()->grad_[0], 2.0);
  EXPECT_EQ(a.get()->grad_[1], 3.0);
  EXPECT_EQ(a.get()->grad_[2], 4.0);
  EXPECT_EQ(a.get()->grad_[3], 5.0);

  EXPECT_EQ(b.get()->grad_[0], 5.0);
  EXPECT_EQ(b.get()->grad_[1], 4.0);
  EXPECT_EQ(b.get()->grad_[2], 3.0);
  EXPECT_EQ(b.get()->grad_[3], 2.0);
}

TEST(Basic, Divide) {
  /*
  >>> a = torch.tensor((5.0, 4.0, 3.0, 2.0), requires_grad=True)
  >>> b = torch.tensor((2.0, 3.0, 4.0, 5.0), requires_grad=True)
  >>> a
  tensor([5., 4., 3., 2.], requires_grad=True)
  >>> b
  tensor([2., 3., 4., 5.], requires_grad=True)
  >>> c = a / b
  >>> l = c.sum()
  >>> l.backward()
  >>> l
  tensor(4.9833, grad_fn=<SumBackward0>)
  >>> a.grad
  tensor([0.5000, 0.3333, 0.2500, 0.2000])
  >>> b.grad
  tensor([-1.2500, -0.4444, -0.1875, -0.0800])
  */
  auto a = cg::tensor({4}, {5.0, 4.0, 3.0, 2.0});
  auto b = cg::tensor({4}, {2.0, 3.0, 4.0, 5.0});
  auto c = a / b;
  auto l = c.get()->sum();
  l.get()->backward();

  EXPECT_NEAR(l.get()->data_[0], 4.9833, 0.01);
  EXPECT_EQ(l.get()->grad_.size(), 1);

  EXPECT_EQ(a.get()->grad_.size(), 4);
  EXPECT_EQ(b.get()->grad_.size(), 4);

  EXPECT_NEAR(a.get()->grad_[0], 0.5, 0.01);
  EXPECT_NEAR(a.get()->grad_[1], 0.3333, 0.01);
  EXPECT_NEAR(a.get()->grad_[2], 0.25, 0.01);
  EXPECT_NEAR(a.get()->grad_[3], 0.2, 0.01);

  EXPECT_NEAR(b.get()->grad_[0], -1.25, 0.01);
  EXPECT_NEAR(b.get()->grad_[1], -0.444, 0.01);
  EXPECT_NEAR(b.get()->grad_[2], -0.1875, 0.01);
  EXPECT_NEAR(b.get()->grad_[3], -0.08, 0.01);
}

TEST(Basic, MatMul) {
  /*
  >>> a = torch.tensor(((5.0, 4.0), (3.0, 2.0)), requires_grad=True)
  >>> b = torch.tensor(((2.0, 3.0), (4.0, 5.0)), requires_grad=True)
  >>> c = a.matmul(b)
  >>> l = c.sum()
  >>> l.backward()
  >>> l
  tensor(94., grad_fn=<SumBackward0>)
  >>> a.grad
  tensor([[5., 9.],
          [5., 9.]])
  >>> b.grad
  tensor([[8., 8.],
          [6., 6.]])
  */
  auto a = cg::tensor({2, 2}, {5.0, 4.0, 3.0, 2.0});
  auto b = cg::tensor({2, 2}, {2.0, 3.0, 4.0, 5.0});
  auto c = a.get()->matmul(b);
  auto l = c.get()->sum();
  l.get()->backward();

  EXPECT_NEAR(l.get()->data_[0], 94.0, 0.01);
  EXPECT_EQ(l.get()->grad_.size(), 1);

  EXPECT_EQ(a.get()->grad_.size(), 4);
  EXPECT_EQ(b.get()->grad_.size(), 4);

  EXPECT_NEAR(a.get()->grad_[0], 5.0, 0.01);
  EXPECT_NEAR(a.get()->grad_[1], 9.0, 0.01);
  EXPECT_NEAR(a.get()->grad_[2], 5.0, 0.01);
  EXPECT_NEAR(a.get()->grad_[3], 9.0, 0.01);

  EXPECT_NEAR(b.get()->grad_[0], 8.0, 0.01);
  EXPECT_NEAR(b.get()->grad_[1], 8.0, 0.01);
  EXPECT_NEAR(b.get()->grad_[2], 6.0, 0.01);
  EXPECT_NEAR(b.get()->grad_[3], 6.0, 0.01);
}

TEST(Basic, MatMulAdd) {
  /*
>>> a = torch.tensor(((5.0, 4.0), (3.0, 2.0)), requires_grad=True)
>>> b = torch.tensor( [[3.0], [4.0]], requires_grad=True)
>>> c = torch.tensor([[6.0], [7.0]], requires_grad=True)
>>> a
tensor([[5., 4.],
        [3., 2.]], requires_grad=True)
>>> b
tensor([[3.],
        [4.]], requires_grad=True)
>>> c
tensor([[6.],
        [7.]], requires_grad=True)
>>> a @ b
tensor([[31.],
        [17.]], grad_fn=<MmBackward0>)
>>> (a @ b) + c
tensor([[37.],
        [24.]], grad_fn=<AddBackward0>)
>>> d = (a @ b) + c
>>> l = d.sum()
>>> l.backward()
>>> a.grad
tensor([[3., 4.],
        [3., 4.]])
>>> b.grad
tensor([[8.],
        [6.]])
>>> c.grad
tensor([[1.],
        [1.]])
>>> d
tensor([[37.],
        [24.]], grad_fn=<AddBackward0>)
>>> l
tensor(61., grad_fn=<SumBackward0>)
  */
  auto a = cg::tensor({2, 2}, {5.0, 4.0, 3.0, 2.0});
  auto b = cg::tensor({2, 1}, {3.0, 4.0});
  auto c = cg::tensor({2, 1}, {6.0, 7.0});
  auto d = (a.get()->matmul(b) + c);
  auto l = d.get()->sum();
  l.get()->backward();

  EXPECT_EQ(l.get()->grad_.size(), 1);
  EXPECT_NEAR(l.get()->data_[0], 61.0, 0.01);

  EXPECT_EQ(a.get()->grad_.size(), 4);
  EXPECT_NEAR(a.get()->grad_[0], 3.0, 0.01);
  EXPECT_NEAR(a.get()->grad_[1], 4.0, 0.01);
  EXPECT_NEAR(a.get()->grad_[2], 3.0, 0.01);
  EXPECT_NEAR(a.get()->grad_[3], 4.0, 0.01);

  EXPECT_EQ(b.get()->grad_.size(), 2);
  EXPECT_NEAR(b.get()->grad_[0], 8.0, 0.01);
  EXPECT_NEAR(b.get()->grad_[1], 6.0, 0.01);

  EXPECT_EQ(c.get()->grad_.size(), 2);
  EXPECT_NEAR(c.get()->grad_[0], 1.0, 0.01);
  EXPECT_NEAR(c.get()->grad_[1], 1.0, 0.01);
}

TEST(Basic, MatMulAddSigmoid) {
  /*
  >>> a = torch.tensor(((0.1, 0.2), (-0.3, 0.4)), requires_grad=True)
  >>> b = torch.tensor( [[0.5], [0.6]], requires_grad=True)
  >>> c = torch.tensor([[0.8], [0.9]], requires_grad=True)
  >>> a
  tensor([[ 0.1000,  0.2000],
          [-0.3000,  0.4000]], requires_grad=True)
  >>> b
  tensor([[0.5000],
          [0.6000]], requires_grad=True)
  >>> c
  tensor([[0.8000],
          [0.9000]], requires_grad=True)
  >>> a @ b
  tensor([[0.1700],
          [0.0900]], grad_fn=<MmBackward0>)
  >>> (a @ b) + c
  tensor([[0.9700],
          [0.9900]], grad_fn=<AddBackward0>)
  >>> torch.sigmoid((a @ b) + c)
  tensor([[0.7251],
          [0.7291]], grad_fn=<SigmoidBackward0>)
  >>> l = torch.sigmoid((a @ b) + c).sum()
  >>> l
  tensor(1.4542, grad_fn=<SumBackward0>)
  >>> l.backward()
  >>> a.grad
  tensor([[0.0997, 0.1196],
          [0.0988, 0.1185]])
  >>> b.grad
  tensor([[-0.0393],
          [ 0.1189]])
  >>> c.grad
  tensor([[0.1993],
          [0.1975]])
  */
  auto a = cg::tensor({2, 2}, {0.1, 0.2, -0.3, 0.4});
  auto b = cg::tensor({2, 1}, {0.5, 0.6});
  auto c = cg::tensor({2, 1}, {0.8, 0.9});
  auto l = (a.get()->matmul(b) + c).get()->sigmoid().get()->sum();
  l.get()->backward();

  EXPECT_EQ(l.get()->grad_.size(), 1);
  EXPECT_NEAR(l.get()->data_[0], 1.4542, 0.01);

  EXPECT_EQ(a.get()->grad_.size(), 4);
  EXPECT_NEAR(a.get()->grad_[0], 0.0997, 0.01);
  EXPECT_NEAR(a.get()->grad_[1], 0.1196, 0.01);
  EXPECT_NEAR(a.get()->grad_[2], 0.0988, 0.01);
  EXPECT_NEAR(a.get()->grad_[3], 0.1185, 0.01);

  EXPECT_EQ(b.get()->grad_.size(), 2);
  EXPECT_NEAR(b.get()->grad_[0], -0.0393, 0.01);
  EXPECT_NEAR(b.get()->grad_[1], 0.1189, 0.01);

  EXPECT_EQ(c.get()->grad_.size(), 2);
  EXPECT_NEAR(c.get()->grad_[0], 0.1993, 0.01);
  EXPECT_NEAR(c.get()->grad_[1], 0.1975, 0.01);
}

TEST(Basic, ChainedMM) {
  /*
  >>> import torch
  >>> a = torch.tensor(((5.0, 4.0), (3.0, 2.0)), requires_grad=True)
  >>> b = torch.tensor(((2.0, 3.0), (4.0, 5.0)), requires_grad=True)
  >>> c = a.matmul(b)
  >>> d = c.matmul(b)
  >>> l = d.sum()
  >>> l
  tensor(686., grad_fn=<SumBackward0>)
  >>> l.backward()
  >>> l
  tensor(686., grad_fn=<SumBackward0>)
  >>> d
  tensor([[192., 253.],
          [104., 137.]], grad_fn=<MmBackward0>)
  >>> c
  tensor([[26., 35.],
          [14., 19.]], grad_fn=<MmBackward0>)
  >>> a.grad
  tensor([[37., 65.],
          [37., 65.]])
  >>> b.grad
  tensor([[ 80., 112.],
          [ 84., 108.]])
  */
  auto a = cg::tensor({2, 2}, {5.0, 4.0, 3.0, 2.0});
  auto b = cg::tensor({2, 2}, {2.0, 3.0, 4.0, 5.0});
  auto c = a.get()->matmul(b);
  auto d = c.get()->matmul(b);
  auto l = d.get()->sum();
  l.get()->backward();

  EXPECT_NEAR(l.get()->data_[0], 686.0, 0.01);
  EXPECT_EQ(l.get()->grad_.size(), 1);

  EXPECT_EQ(a.get()->grad_.size(), 4);
  EXPECT_EQ(b.get()->grad_.size(), 4);

  EXPECT_NEAR(a.get()->grad_[0], 37.0, 0.01);
  EXPECT_NEAR(a.get()->grad_[1], 65.0, 0.01);
  EXPECT_NEAR(a.get()->grad_[2], 37.0, 0.01);
  EXPECT_NEAR(a.get()->grad_[3], 65.0, 0.01);

  EXPECT_NEAR(b.get()->grad_[0], 80.0, 0.01);
  EXPECT_NEAR(b.get()->grad_[1], 112.0, 0.01);
  EXPECT_NEAR(b.get()->grad_[2], 84.0, 0.01);
  EXPECT_NEAR(b.get()->grad_[3], 108.0, 0.01);
}

TEST(Basic, ScalarComplexAB) {
  /*
  >>> import torch
  >>> a = torch.tensor((2.0), requires_grad=True)
  >>> b = torch.tensor((43.0), requires_grad=True)
  >>> c = (a * b) - ((b / a) + b)
  >>> c.backward()
  >>> c
  tensor(21.5000, grad_fn=<SubBackward0>)
  >>> a.grad
  tensor(53.7500)
  >>> b.grad
  tensor(0.5000)
  */
  auto a = cg::tensor({1}, {2.0});
  auto b = cg::tensor({1}, {43.0});
  auto l = (a * b) - ((b / a) + b);  // NOTE im calling this l not c lazy
  l.get()->backward();

  EXPECT_NEAR(l.get()->data_[0], 21.5, 0.01);
  EXPECT_EQ(l.get()->grad_.size(), 1);

  EXPECT_EQ(a.get()->grad_.size(), 1);
  EXPECT_EQ(b.get()->grad_.size(), 1);

  EXPECT_NEAR(a.get()->grad_[0], 53.7500, 0.01);

  EXPECT_NEAR(b.get()->grad_[0], 0.5000, 0.01);
}

TEST(Basic, ChainedComplexOperations) {
  /*
  >>> import torch
  >>> a = torch.tensor(((2.0, 3.0), (4.0, 5.0)), requires_grad=True)
  >>> b = torch.tensor(((6.0, 7.0), (8.0, 9.0)), requires_grad=True)
  >>> c = torch.tensor(((10.0, 10.0), (10.0, 10.0)), requires_grad=True)
  >>> d = torch.tensor(((11.0, 11.0), (11.0, 11.0)), requires_grad=True)
  >>> e = (a.matmul(b) + c) * d
  >>> f = e.sum()
  >>> f.backward()
  >>> a.grad
  tensor([[143., 187.],
          [143., 187.]])
  >>> b.grad
  tensor([[66., 66.],
          [88., 88.]])
  >>> c.grad
  tensor([[11., 11.],
          [11., 11.]])
  >>> d.grad
  tensor([[46., 51.],
          [74., 83.]])
  >>> f
  tensor(2794., grad_fn=<SumBackward0>)
  >>> f
  tensor(2794., grad_fn=<SumBackward0>)
  */
  auto a = cg::tensor({2, 2}, {2.0, 3.0, 4.0, 5.0});
  auto b = cg::tensor({2, 2}, {6.0, 7.0, 8.0, 9.0});
  auto c = cg::tensor({2, 2}, {10.0, 10.0, 10.0, 10.0});
  auto d = cg::tensor({2, 2}, {11.0, 11.0, 11.0, 11.0});
  auto e = (a.get()->matmul(b) + c) * d;
  auto f = e.get()->sum();
  f.get()->backward();

  EXPECT_NEAR(f.get()->data_[0], 2794.0, 0.01);
  EXPECT_EQ(f.get()->grad_.size(), 1);

  EXPECT_EQ(a.get()->grad_.size(), 4);
  EXPECT_EQ(b.get()->grad_.size(), 4);
  EXPECT_EQ(c.get()->grad_.size(), 4);
  EXPECT_EQ(d.get()->grad_.size(), 4);

  EXPECT_NEAR(a.get()->grad_[0], 143.0, 0.01);
  EXPECT_NEAR(a.get()->grad_[1], 187.0, 0.01);
  EXPECT_NEAR(a.get()->grad_[2], 143.0, 0.01);
  EXPECT_NEAR(a.get()->grad_[3], 187.0, 0.01);

  EXPECT_NEAR(b.get()->grad_[0], 66.0, 0.01);
  EXPECT_NEAR(b.get()->grad_[1], 66.0, 0.01);
  EXPECT_NEAR(b.get()->grad_[2], 88.0, 0.01);
  EXPECT_NEAR(b.get()->grad_[3], 88.0, 0.01);

  EXPECT_NEAR(c.get()->grad_[0], 11.0, 0.01);
  EXPECT_NEAR(c.get()->grad_[1], 11.0, 0.01);
  EXPECT_NEAR(c.get()->grad_[2], 11.0, 0.01);
  EXPECT_NEAR(c.get()->grad_[3], 11.0, 0.01);

  EXPECT_NEAR(d.get()->grad_[0], 46.0, 0.01);
  EXPECT_NEAR(d.get()->grad_[1], 51.0, 0.01);
  EXPECT_NEAR(d.get()->grad_[2], 74.0, 0.01);
  EXPECT_NEAR(d.get()->grad_[3], 83.0, 0.01);
}

TEST(Basic, ReLU) {
  /*
  >>> import torch
  >>> a = torch.tensor(((-1.0, -2.0), (1.0, 2.0)), requires_grad=True)
  >>> b = a.relu()
  >>> b
  tensor([[0., 0.],
          [1., 2.]], grad_fn=<ReluBackward0>)
  >>> l = b.sum()
  >>> l.backward()
  >>> l
  tensor(3., grad_fn=<SumBackward0>)
  >>> a.grad
  tensor([[0., 0.],
          [1., 1.]])
  */
  auto a = cg::tensor({2, 2}, {-1.0, -2.0, 1.0, 2.0});
  auto b = a.get()->relu();
  auto l = b.get()->sum();
  l.get()->backward();

  EXPECT_EQ(l.get()->data_[0], 3.0);
  EXPECT_EQ(l.get()->grad_.size(), 1);

  EXPECT_EQ(b.get()->data_.size(), 4);
  EXPECT_EQ(b.get()->data_[0], 0.0);
  EXPECT_EQ(b.get()->data_[1], 0.0);
  EXPECT_EQ(b.get()->data_[2], 1.0);
  EXPECT_EQ(b.get()->data_[3], 2.0);

  EXPECT_EQ(a.get()->grad_.size(), 4);
  EXPECT_EQ(a.get()->grad_[0], 0.0);
  EXPECT_EQ(a.get()->grad_[1], 0.0);
  EXPECT_EQ(a.get()->grad_[2], 1.0);
  EXPECT_EQ(a.get()->grad_[3], 1.0);
}

TEST(Basic, Sigmoid) {
  /*
  >>> import torch
  >>> torch.__version__
  '2.3.0a0+gitbbded92'
  >>> a = torch.tensor(((-1.0, -2.0), (1.0, 2.0)), requires_grad=True)
  >>> b = a.sigmoid()
  >>> b
  tensor([[0.2689, 0.1192],
          [0.7311, 0.8808]], grad_fn=<SigmoidBackward0>)
  >>> l = b.sum()
  >>> l.backward()
  >>> l
  tensor(2., grad_fn=<SumBackward0>)
  >>> a.grad
  tensor([[0.1966, 0.1050],
          [0.1966, 0.1050]])
  */
  auto a = cg::tensor({2, 2}, {-1.0, -2.0, 1.0, 2.0});
  auto b = a.get()->sigmoid();
  auto l = b.get()->sum();
  l.get()->backward();

  EXPECT_EQ(l.get()->data_[0], 2.0);
  EXPECT_EQ(l.get()->grad_.size(), 1);

  EXPECT_EQ(b.get()->data_.size(), 4);
  EXPECT_NEAR(b.get()->data_[0], 0.2689, 0.01);
  EXPECT_NEAR(b.get()->data_[1], 0.1192, 0.01);
  EXPECT_NEAR(b.get()->data_[2], 0.7311, 0.01);
  EXPECT_NEAR(b.get()->data_[3], 0.8808, 0.01);

  EXPECT_EQ(a.get()->grad_.size(), 4);
  EXPECT_NEAR(a.get()->grad_[0], 0.1966, 0.01);
  EXPECT_NEAR(a.get()->grad_[1], 0.1050, 0.01);
  EXPECT_NEAR(a.get()->grad_[2], 0.1966, 0.01);
  EXPECT_NEAR(a.get()->grad_[3], 0.1050, 0.01);
}

TEST(Basic, Exp) {
  /*
  >>> a = torch.tensor([0, 1, 2, 3.], requires_grad=True)
  >>> a
  tensor([0., 1., 2., 3.], requires_grad=True)
  >>> b = a.exp()
  >>>
  >>> b
  tensor([ 1.0000,  2.7183,  7.3891, 20.0855], grad_fn=<ExpBackward0>)
  >>> l = b.sum()
  >>> l
  tensor(31.1929, grad_fn=<SumBackward0>)
  >>> l.backward()
  >>> a.grad
  tensor([ 1.0000,  2.7183,  7.3891, 20.0855])
  */
  auto a = cg::tensor({2, 2}, {0.0, 1.0, 2.0, 3.0});
  auto b = a.get()->exponential();
  auto l = b.get()->sum();
  l.get()->backward();

  EXPECT_NEAR(l.get()->data_[0], 31.1929, 0.01);
  EXPECT_EQ(l.get()->grad_.size(), 1);

  EXPECT_EQ(b.get()->data_.size(), 4);
  EXPECT_NEAR(b.get()->data_[0], 1.0000, 0.01);
  EXPECT_NEAR(b.get()->data_[1], 2.7183, 0.01);
  EXPECT_NEAR(b.get()->data_[2], 7.3891, 0.01);
  EXPECT_NEAR(b.get()->data_[3], 20.0855, 0.01);

  EXPECT_EQ(a.get()->grad_.size(), 4);
  EXPECT_NEAR(a.get()->grad_[0], 1.0000, 0.01);
  EXPECT_NEAR(a.get()->grad_[1], 2.7183, 0.01);
  EXPECT_NEAR(a.get()->grad_[2], 7.3891, 0.01);
  EXPECT_NEAR(a.get()->grad_[3], 20.0855, 0.01);
}

TEST(Basic, Ln) {
  /*
  >>> a = torch.tensor([1, 2.], requires_grad=True)
  >>> l = a.log().sum()
  >>> l.backward()
  >>> a.grad
  tensor([1.0000, 0.5000])
  >>> l
  tensor(0.6931, grad_fn=<SumBackward0>)
  >>> a
  tensor([1., 2.], requires_grad=True)
  >>> l
  tensor(0.6931, grad_fn=<SumBackward0>)
  >>> l.grad
  <stdin>:1: UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). If you indeed want the .grad field to be populated for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. See github.com/pytorch/pytorch/pull/30531 for more informations. (Triggered internally at aten/src/ATen/core/TensorBody.h:489.)
  */
  auto a = cg::tensor({2}, {1.0, 2.0});
  auto b = a.get()->ln();
  auto l = b.get()->sum();
  l.get()->backward();

  EXPECT_NEAR(l.get()->data_[0], 0.6931, 0.01);
  EXPECT_EQ(l.get()->grad_.size(), 1);

  EXPECT_EQ(a.get()->grad_.size(), 2);
  EXPECT_NEAR(a.get()->grad_[0], 1.0, 0.01);
  EXPECT_NEAR(a.get()->grad_[1], 0.5, 0.01);
}

TEST(Broadcast, Divide) {
  /*
  >>> import torch
  >>> t = torch.tensor((5.0, 4.0, -3.0, 2.0), requires_grad=True)
  >>> t
  tensor([ 5.,  4., -3.,  2.], requires_grad=True)
  >>> t / torch.tensor([2.], requires_grad=True)
  tensor([ 2.5000,  2.0000, -1.5000,  1.0000], grad_fn=<DivBackward0>)
  */
  auto a = cg::tensor({4}, {5.0, 4.0, 3.0, 2.0});
  auto b = cg::tensor({4}, {2.0, 3.0, 4.0, 5.0});
  auto c = a / b;
  auto l = c.get()->sum();
  l.get()->backward();

  EXPECT_NEAR(l.get()->data_[0], 4.9833, 0.01);
  EXPECT_EQ(l.get()->grad_.size(), 1);

  EXPECT_EQ(a.get()->grad_.size(), 4);
  EXPECT_EQ(b.get()->grad_.size(), 4);

  EXPECT_NEAR(a.get()->grad_[0], 0.5, 0.01);
  EXPECT_NEAR(a.get()->grad_[1], 0.3333, 0.01);
  EXPECT_NEAR(a.get()->grad_[2], 0.25, 0.01);
  EXPECT_NEAR(a.get()->grad_[3], 0.2, 0.01);

  EXPECT_NEAR(b.get()->grad_[0], -1.25, 0.01);
  EXPECT_NEAR(b.get()->grad_[1], -0.444, 0.01);
  EXPECT_NEAR(b.get()->grad_[2], -0.1875, 0.01);
  EXPECT_NEAR(b.get()->grad_[3], -0.08, 0.01);
}

TEST(SigmoidGauntlet, MatmulAddSigmoid0) {
  // Tensor.sigmoid(w0 @ x + b0)

  // >>> import torch
  // >>> w0 = torch.tensor([[-0.5, -0.5], [0.5, -0.5]], requires_grad=True)
  // >>> x = torch.tensor([[1.0], [1.0]], requires_grad=True)
  // >>> b0 = torch.tensor([[-0.5], [-0.5]], requires_grad=True)
  // >>> s = torch.sigmoid(w0 @ x + b0)
  // >>> s
  // tensor([[0.1824],
  //         [0.3775]], grad_fn=<SigmoidBackward0>)
  // >>> l = s.sum()
  // >>> l
  // tensor(0.5600, grad_fn=<SumBackward0>)
  // >>> l.backward()
  // >>> w0.grad
  // tensor([[0.1491, 0.1491],
  //         [0.2350, 0.2350]])
  // >>> x.grad
  // tensor([[ 0.0429],
  //         [-0.1921]])
  // >>> b0.grad
  // tensor([[0.1491],
  //         [0.2350]])

  auto w0 = cg::tensor({2, 2}, {-0.5, -0.5, 0.5, -0.5});
  auto x = cg::tensor({2, 1}, {1.0, 1.0});
  auto b0 = cg::tensor({2, 1}, {-0.5, -0.5});
  auto muldot = w0.get()->matmul(x) + b0;
  auto s = muldot.get()->sigmoid();
  auto l = s.get()->sum();
  l.get()->backward();

  EXPECT_EQ(w0.get()->grad_.size(), 4);

  EXPECT_NEAR(l.get()->data_[0], 0.5600, 0.01);

  EXPECT_NEAR(s.get()->data_[0], 0.1824, 0.01);
  EXPECT_NEAR(s.get()->data_[1], 0.3775, 0.01);

  EXPECT_NEAR(w0.get()->grad_[0], 0.1491, 0.01);
  EXPECT_NEAR(w0.get()->grad_[1], 0.1491, 0.01);
  EXPECT_NEAR(w0.get()->grad_[2], 0.2350, 0.01);
  EXPECT_NEAR(w0.get()->grad_[3], 0.2350, 0.01);

  EXPECT_NEAR(x.get()->grad_[0], 0.0429, 0.01);
  EXPECT_NEAR(x.get()->grad_[1], -0.1921, 0.01);

  EXPECT_NEAR(b0.get()->grad_[0], 0.1491, 0.01);
  EXPECT_NEAR(b0.get()->grad_[1], 0.2350, 0.01);
}

TEST(SigmoidGauntlet, MatmulAddSigmoid1) {
  // Tensor.sigmoid(w0 @ x + b0)

  // >>> import torch
  // >>> w0 = torch.tensor([[0.5, 0.5], [0.5, 0.5]], requires_grad=True)
  // >>> x = torch.tensor([[1.0], [1.0]], requires_grad=True)
  // >>> b0 = torch.tensor([[0.5], [0.5]], requires_grad=True)
  // >>> s = torch.sigmoid(w0 @ x + b0)
  // >>> s
  // tensor([[0.8176],
  //         [0.8176]], grad_fn=<SigmoidBackward0>)
  // >>> l = s.sum()
  // >>> l
  // tensor(1.6351, grad_fn=<SumBackward0>)
  // >>> l.backward()
  // >>> w0.grad
  // tensor([[0.1491, 0.1491],
  //         [0.1491, 0.1491]])
  // >>> x.grad
  // tensor([[0.1491],
  //         [0.1491]])
  // >>> b0.grad
  // tensor([[0.1491],
  //         [0.1491]])

  auto w0 = cg::tensor({2, 2}, {0.5, 0.5, 0.5, 0.5});
  auto x = cg::tensor({2, 1}, {1.0, 1.0});
  auto b0 = cg::tensor({2, 1}, {0.5, 0.5});
  auto muldot = w0.get()->matmul(x) + b0;
  auto s = muldot.get()->sigmoid();
  auto l = s.get()->sum();
  l.get()->backward();

  EXPECT_EQ(w0.get()->grad_.size(), 4);

  EXPECT_NEAR(l.get()->data_[0], 1.6351, 0.01);

  EXPECT_NEAR(s.get()->data_[0], 0.8176, 0.01);
  EXPECT_NEAR(s.get()->data_[1], 0.8176, 0.01);

  EXPECT_NEAR(w0.get()->grad_[0], 0.1491, 0.01);
  EXPECT_NEAR(w0.get()->grad_[1], 0.1491, 0.01);
  EXPECT_NEAR(w0.get()->grad_[2], 0.1491, 0.01);
  EXPECT_NEAR(w0.get()->grad_[3], 0.1491, 0.01);

  EXPECT_NEAR(x.get()->grad_[0], 0.1491, 0.01);
  EXPECT_NEAR(x.get()->grad_[1], 0.1491, 0.01);

  EXPECT_NEAR(b0.get()->grad_[0], 0.1491, 0.01);
  EXPECT_NEAR(b0.get()->grad_[1], 0.1491, 0.01);
}

TEST(SigmoidGauntlet, MatmulAddSigmoid2) {
  // Tensor.sigmoid(w0 @ x + b0)

  // >>> import torch
  // >>> w0 = torch.tensor([[-0.5, 0.5], [0.5, 0.5]], requires_grad=True)
  // >>> x = torch.tensor([[1.0], [1.0]], requires_grad=True)
  // >>> b0 = torch.tensor([[0.5], [0.5]], requires_grad=True)
  // >>> s = torch.sigmoid(w0 @ x + b0)
  // >>> s
  // tensor([[0.6225],
  //         [0.8176]], grad_fn=<SigmoidBackward0>)
  // >>> l = s.sum()
  // >>> l
  // tensor(1.4400, grad_fn=<SumBackward0>)
  // >>> l.backward()
  // >>> w0.grad
  // tensor([[0.2350, 0.2350],
  //         [0.1491, 0.1491]])
  // >>> x.grad
  // tensor([[-0.0429],
  //         [ 0.1921]])
  // >>> b0.grad
  // tensor([[0.2350],
  //         [0.1491]])

  auto w0 = cg::tensor({2, 2}, {-0.5, 0.5, 0.5, 0.5});
  auto x = cg::tensor({2, 1}, {1.0, 1.0});
  auto b0 = cg::tensor({2, 1}, {0.5, 0.5});
  auto muldot = w0.get()->matmul(x) + b0;
  auto s = muldot.get()->sigmoid();
  auto l = s.get()->sum();
  l.get()->backward();

  EXPECT_EQ(w0.get()->grad_.size(), 4);

  EXPECT_NEAR(l.get()->data_[0], 1.4400, 0.01);

  EXPECT_NEAR(s.get()->data_[0], 0.6225, 0.01);
  EXPECT_NEAR(s.get()->data_[1], 0.8176, 0.01);

  EXPECT_NEAR(w0.get()->grad_[0], 0.2350, 0.01);
  EXPECT_NEAR(w0.get()->grad_[1], 0.2350, 0.01);
  EXPECT_NEAR(w0.get()->grad_[2], 0.1491, 0.01);
  EXPECT_NEAR(w0.get()->grad_[3], 0.1491, 0.01);

  EXPECT_NEAR(x.get()->grad_[0], -0.0429, 0.01);
  EXPECT_NEAR(x.get()->grad_[1], 0.1921, 0.01);

  EXPECT_NEAR(b0.get()->grad_[0], 0.2350, 0.01);
  EXPECT_NEAR(b0.get()->grad_[1], 0.1491, 0.01);
}

TEST(SigmoidGauntlet, MatmulAddSigmoid3) {
  // Tensor.sigmoid(w0 @ x + b0)

  // >>> import torch
  // >>> w0 = torch.tensor([[-0.5, 0.5], [0.5, 0.5]], requires_grad=True)
  // >>> x = torch.tensor([[1.0], [1.0]], requires_grad=True)
  // >>> b0 = torch.tensor([[1.0], [1.0]], requires_grad=True)
  // >>> s = torch.sigmoid(w0 @ x + b0)
  // >>> s
  // tensor([[0.7311],
  //         [0.8808]], grad_fn=<SigmoidBackward0>)
  // >>> l = s.sum()
  // >>> l
  // tensor(1.6119, grad_fn=<SumBackward0>)
  // >>> l.backward()
  // >>> w0.grad
  // tensor([[0.1966, 0.1966],
  //         [0.1050, 0.1050]])
  // >>> x.grad
  // tensor([[-0.0458],
  //         [ 0.1508]])
  // >>> b0.grad
  // tensor([[0.1966],
  //         [0.1050]])

  auto w0 = cg::tensor({2, 2}, {-0.5, 0.5, 0.5, 0.5});
  auto x = cg::tensor({2, 1}, {1.0, 1.0});
  auto b0 = cg::tensor({2, 1}, {1.0, 1.0});
  auto muldot = w0.get()->matmul(x) + b0;
  auto s = muldot.get()->sigmoid();
  auto l = s.get()->sum();
  l.get()->backward();

  EXPECT_EQ(w0.get()->grad_.size(), 4);

  EXPECT_NEAR(l.get()->data_[0], 1.6119, 0.01);

  EXPECT_NEAR(s.get()->data_[0], 0.7311, 0.01);
  EXPECT_NEAR(s.get()->data_[1], 0.8808, 0.01);

  EXPECT_NEAR(w0.get()->grad_[0], 0.1966, 0.01);
  EXPECT_NEAR(w0.get()->grad_[1], 0.1966, 0.01);
  EXPECT_NEAR(w0.get()->grad_[2], 0.1050, 0.01);
  EXPECT_NEAR(w0.get()->grad_[3], 0.1050, 0.01);

  EXPECT_NEAR(x.get()->grad_[0], -0.0458, 0.01);
  EXPECT_NEAR(x.get()->grad_[1], 0.1508, 0.01);

  EXPECT_NEAR(b0.get()->grad_[0], 0.1966, 0.01);
  EXPECT_NEAR(b0.get()->grad_[1], 0.1050, 0.01);
}

TEST(SigmoidGauntlet, MatmulAddSigmoid4) {
  // Tensor.sigmoid(w0 @ x + b0)

  // >>> import torch
  // >>> w0 = torch.tensor([[-1.0, 1.0], [1.0, 1.0]], requires_grad=True)
  // >>> x = torch.tensor([[1.0], [1.0]], requires_grad=True)
  // >>> b0 = torch.tensor([[1.0], [1.0]], requires_grad=True)
  // >>> s = torch.sigmoid(w0 @ x + b0)
  // >>> s
  // tensor([[0.7311],
  //         [0.9526]], grad_fn=<SigmoidBackward0>)
  // >>> l = s.sum()
  // >>> l
  // tensor(1.6836, grad_fn=<SumBackward0>)
  // >>> w0.grad
  // >>> l.backward()
  // >>> w0.grad
  // tensor([[0.1966, 0.1966],
  //         [0.0452, 0.0452]])
  // >>> x.grad
  // tensor([[-0.1514],
  //         [ 0.2418]])
  // >>> b0.grad
  // tensor([[0.1966],
  //         [0.0452]])

  auto w0 = cg::tensor({2, 2}, {-1.0, 1.0, 1.0, 1.0});
  auto x = cg::tensor({2, 1}, {1.0, 1.0});
  auto b0 = cg::tensor({2, 1}, {1.0, 1.0});
  auto muldot = w0.get()->matmul(x) + b0;
  auto s = muldot.get()->sigmoid();
  auto l = s.get()->sum();
  l.get()->backward();

  EXPECT_EQ(w0.get()->grad_.size(), 4);

  EXPECT_NEAR(l.get()->data_[0], 1.6836, 0.01);

  EXPECT_NEAR(s.get()->data_[0], 0.7311, 0.01);
  EXPECT_NEAR(s.get()->data_[1], 0.9526, 0.01);

  EXPECT_NEAR(w0.get()->grad_[0], 0.1966, 0.01);
  EXPECT_NEAR(w0.get()->grad_[1], 0.1966, 0.01);
  EXPECT_NEAR(w0.get()->grad_[2], 0.0452, 0.01);
  EXPECT_NEAR(w0.get()->grad_[3], 0.0452, 0.01);

  EXPECT_NEAR(x.get()->grad_[0], -0.1514, 0.01);
  EXPECT_NEAR(x.get()->grad_[1], 0.2418, 0.01);

  EXPECT_NEAR(b0.get()->grad_[0], 0.1966, 0.01);
  EXPECT_NEAR(b0.get()->grad_[1], 0.0452, 0.01);
}

TEST(SigmoidGauntlet, MatmulAddSigmoid5) {
  // Tensor.sigmoid(w0 @ x + b0)

  // >>> import torch
  // >>> w0 = torch.tensor([[1.0, 1.0], [1.0, 1.0]], requires_grad=True)
  // >>> x = torch.tensor([[1.0], [1.0]], requires_grad=True)
  // >>> b0 = torch.tensor([[1.0], [1.0]], requires_grad=True)
  // >>> s = torch.sigmoid(w0 @ x + b0)
  // >>> s
  // tensor([[0.9526],
  //         [0.9526]], grad_fn=<SigmoidBackward0>)
  // >>> l = s.sum()
  // >>> l
  // tensor(1.9051, grad_fn=<SumBackward0>)
  // >>> l.backward()
  // >>> w0.grad
  // tensor([[0.0452, 0.0452],
  //         [0.0452, 0.0452]])
  // >>> x.grad
  // tensor([[0.0904],
  //         [0.0904]])
  // >>> b0.grad
  // tensor([[0.0452],
  //         [0.0452]])

  auto w0 = cg::tensor({2, 2}, {1.0, 1.0, 1.0, 1.0});
  auto x = cg::tensor({2, 1}, {1.0, 1.0});
  auto b0 = cg::tensor({2, 1}, {1.0, 1.0});
  auto muldot = w0.get()->matmul(x) + b0;
  auto s = muldot.get()->sigmoid();
  auto l = s.get()->sum();
  l.get()->backward();

  EXPECT_EQ(w0.get()->grad_.size(), 4);

  EXPECT_NEAR(l.get()->data_[0], 1.9051, 0.01);

  EXPECT_NEAR(s.get()->data_[0], 0.9526, 0.01);
  EXPECT_NEAR(s.get()->data_[1], 0.9526, 0.01);

  EXPECT_NEAR(w0.get()->grad_[0], 0.0452, 0.01);
  EXPECT_NEAR(w0.get()->grad_[1], 0.0452, 0.01);
  EXPECT_NEAR(w0.get()->grad_[2], 0.0452, 0.01);
  EXPECT_NEAR(w0.get()->grad_[3], 0.0452, 0.01);

  EXPECT_NEAR(x.get()->grad_[0], 0.0904, 0.01);
  EXPECT_NEAR(x.get()->grad_[1], 0.0904, 0.01);

  EXPECT_NEAR(b0.get()->grad_[0], 0.0452, 0.01);
  EXPECT_NEAR(b0.get()->grad_[1], 0.0452, 0.01);
}

TEST(MLP, InnerMatmul) {
  // w0 @ x

  // >>> w0 = tensor([[-0.5963, -0.0062], [0.1741, -0.1097]],
  // requires_grad=True)
  // >>> x = tensor([[1.0], [1.0]], requires_grad=True)
  // >>> l = (w0 @ x).sum()
  // >>> w0 @ x
  // tensor([[-0.6025],
  //         [ 0.0644]], grad_fn=<MmBackward0>)
  // >>> l.backward()
  // >>> l
  // tensor(-0.5381, grad_fn=<SumBackward0>)
  // >>> w0.grad
  // tensor([[1., 1.],
  //         [1., 1.]])
  // >>> x.grad
  // tensor([[-0.4222],
  //         [-0.1159]])

  auto w0 = cg::tensor({2, 2}, {-0.5963, -0.0062, 0.1741, -0.1097});
  auto x = cg::tensor({2, 1}, {1.0, 1.0});
  auto l = (w0.get()->matmul(x)).get()->sum();
  l.get()->backward();

  EXPECT_EQ(l.get()->data_.size(), 1);
  EXPECT_NEAR(l.get()->data_[0], -0.5381, 0.01);

  EXPECT_EQ(w0.get()->grad_.size(), 4);
  EXPECT_NEAR(w0.get()->grad_[0], 1.0, 0.01);
  EXPECT_NEAR(w0.get()->grad_[1], 1.0, 0.01);
  EXPECT_NEAR(w0.get()->grad_[2], 1.0, 0.01);
  EXPECT_NEAR(w0.get()->grad_[3], 1.0, 0.01);
}

TEST(MLP, InnerNeuron) {
  // (w0 @ x + b0)

  // >>> from torch import tensor
  // >>> w0 = tensor([[-0.5963, -0.0062], [0.1741, -0.1097]],
  // requires_grad=True)
  // >>> x
  // Traceback (most recent call last):
  //   File "<stdin>", line 1, in <module>
  // NameError: name 'x' is not defined
  // >>> x = tensor([[1.0], [1.0]], requires_grad=True)
  // >>> b0 = tensor([[-0.4237], [-0.6666]], requires_grad=True)
  // >>> l = ((w0 @ x) + b0).sum()
  // >>> l.backward()
  // >>> w0
  // tensor([[-0.5963, -0.0062],
  //         [ 0.1741, -0.1097]], requires_grad=True)
  // >>> l
  // tensor(-1.6284, grad_fn=<SumBackward0>)
  // >>> w0.grad
  // tensor([[1., 1.],
  //         [1., 1.]])
  // >>> b0.grad
  // tensor([[1.],
  //         [1.]])

  auto w0 = cg::tensor({2, 2}, {-0.5963, -0.0062, 0.1741, -0.1097});
  auto x = cg::tensor({2, 1}, {1.0, 1.0});
  auto b0 = cg::tensor({2, 1}, {-0.4237, -0.6666});
  auto l = (w0.get()->matmul(x) + b0).get()->sum();
  l.get()->backward();

  EXPECT_EQ(l.get()->data_.size(), 1);
  EXPECT_NEAR(l.get()->data_[0], -1.6284, 0.01);

  EXPECT_EQ(w0.get()->grad_.size(), 4);
  EXPECT_NEAR(w0.get()->grad_[0], 1.0, 0.01);
  EXPECT_NEAR(w0.get()->grad_[1], 1.0, 0.01);
  EXPECT_NEAR(w0.get()->grad_[2], 1.0, 0.01);
  EXPECT_NEAR(w0.get()->grad_[3], 1.0, 0.01);
}

TEST(MLP, InnerSigmoid) {
  // Tensor.sigmoid(w0 @ x + b0)

  // >>> import torch
  // >>> from torch import tensor
  // >>> w0 = torch.tensor([[-0.5963, -0.0062], [0.1741, -0.1097]],
  // requires_grad=True)
  // >>> b0 = torch.tensor([[-0.4237], [-0.6666]], requires_grad=True)
  // >>> x = tensor([[1.0], [1.0]], requires_grad=True)
  // >>> s = torch.sigmoid(w0 @ x + b0)
  // >>> l = s.sum()
  // >>> l.backward()
  // >>> w0
  // tensor([[-0.5963, -0.0062],
  //         [ 0.1741, -0.1097]], requires_grad=True)
  // >>> b0
  // tensor([[-0.4237],
  //         [-0.6666]], requires_grad=True)
  // >>> s
  // tensor([[0.2638],
  //         [0.3538]], grad_fn=<SigmoidBackward0>)
  // >>> l
  // tensor(0.6177, grad_fn=<SumBackward0>)
  // >>> w0.grad
  // tensor([[0.1942, 0.1942],
  //         [0.2286, 0.2286]])
  // >>> b0.grad
  // tensor([[0.1942],
  //         [0.2286]])

  auto w0 = cg::tensor({2, 2}, {-0.5963, -0.0062, 0.1741, -0.1097});
  auto x = cg::tensor({2, 1}, {1.0, 1.0});
  auto b0 = cg::tensor({2}, {-0.4237, -0.6666});
  auto muldot = w0.get()->matmul(x) + b0;
  auto s = muldot.get()->sigmoid();
  auto l = s.get()->sum();
  l.get()->backward();

  EXPECT_EQ(w0.get()->grad_.size(), 4);

  EXPECT_NEAR(l.get()->data_[0], 0.6177, 0.01);

  EXPECT_NEAR(s.get()->data_[0], 0.2638, 0.01);
  EXPECT_NEAR(s.get()->data_[1], 0.3538, 0.01);

  EXPECT_NEAR(b0.get()->grad_[0], 0.1942, 0.01);
  EXPECT_NEAR(b0.get()->grad_[1], 0.2286, 0.01);

  EXPECT_NEAR(w0.get()->grad_[0], 0.1942, 0.01);
  EXPECT_NEAR(w0.get()->grad_[1], 0.1942, 0.01);
  EXPECT_NEAR(w0.get()->grad_[2], 0.2286, 0.01);
  EXPECT_NEAR(w0.get()->grad_[3], 0.2286, 0.01);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
