// Copyright 2023-2024 Ryan Moore
//
// If you remember just one thing out of this course, it should be this – two
// pieces of advice. The first piece of advice is whenever you can, use vector.
// The second piece of advice is if you cannot, find a way how you can. This is
// to avoid any data structures except arrays... Vector is an array, right. Sort
// of, you say, 'Well... Aren't there exceptions?' – Yes, not for you!
//
// Alexander Stepanov

#ifndef SRC_TENSOR_HPP_
#define SRC_TENSOR_HPP_

#ifdef _MSC_VER
#define WEAK
#else
#define WEAK __attribute__((weak))
#endif

#include <pybind11/pybind11.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <fstream>

// #ifdef CUDA_ENABLED
// #include <cuda_runtime.h>
// #endif

template <typename... Args>
void UNUSED(Args &&...args) {
  ([&args]() { (void)args; }(), ...);
}

namespace cg {

namespace py = pybind11;

extern "C" bool cuda_available();

extern "C" const char *helloCPU();
extern "C" const char *helloGPU() WEAK;

const char * hello() {
  if (cuda_available() && helloGPU) {
    return helloGPU();
  } else {
    return helloCPU();
  }
}

// using using for now in case in the future during operator fusion
// I need to know what is actually happening, maybe more clear
struct AutoGradBackward;
struct AddBackward;
struct SubBackward;
struct MulBackward;
struct DivBackward;
using SumBackward = AddBackward;
struct ReluBackward;
struct SigmoidBackward;
struct MatMulBackward;
using SelectBackward = AddBackward;
struct ExpBackward;
struct MaxBackward;
struct ExpBackward;
struct LnBackward;
struct SelectBackward1;
struct SoftmaxBackward;

struct AutoGradForward;
struct MatMulForward;

class Tensor;
class DataProxy;
class GradProxy;

std::shared_ptr<Tensor> tensor(std::initializer_list<size_t>,
                               std::initializer_list<float>);

std::shared_ptr<Tensor> tensor(std::vector<size_t>, std::vector<float>);

std::shared_ptr<Tensor> tensor(std::vector<size_t>, std::vector<float>,
                               std::vector<std::shared_ptr<Tensor>>,
                               std::shared_ptr<AutoGradBackward>, std::string);

const char DEFAULT_BACKWARD_STRING[] = "DefaultBackward";

class Tensor : public std::enable_shared_from_this<Tensor> {
 public:
  std::vector<size_t> size_;
  std::vector<float> data_;
  std::vector<float> grad_;
  std::vector<std::shared_ptr<Tensor>> children_;
  std::shared_ptr<AutoGradBackward> grad_fn_ = nullptr;
  std::string op_ = DEFAULT_BACKWARD_STRING;
  size_t offset_;
  std::vector<size_t> strides_;

  Tensor(std::initializer_list<size_t> size, std::initializer_list<float> data)
      : size_(size),
        data_(data),
        grad_(data.size(), 0),
        children_(),
        grad_fn_(),
        op_(DEFAULT_BACKWARD_STRING),
        offset_(0),
        strides_(size.size(), 0) {
    assert(_product(size) == data_.size());
    _computeStrides();
  }

  Tensor(std::vector<size_t> size, std::vector<float> data)
      : size_(size),
        data_(data),
        grad_(data.size(), 0),
        children_(),
        grad_fn_(),
        op_(DEFAULT_BACKWARD_STRING),
        offset_(0),
        strides_(size.size(), 0) {
    assert(_product(size) == data_.size());
    _computeStrides();
  }

  Tensor(std::vector<size_t> size, std::vector<float> data,
         std::vector<std::shared_ptr<Tensor>> children,
         std::shared_ptr<AutoGradBackward> grad_fn,
         std::string op = DEFAULT_BACKWARD_STRING)
      : size_(size),
        data_(data),
        grad_(data.size(), 0),
        children_(children),
        grad_fn_(std::move(grad_fn)),
        op_(op),
        offset_(0),  // TODO(yrmo) pass as variable
        strides_(size.size(), 0) {
    _computeStrides();
  }

  Tensor(Tensor &&other)
      : size_(std::move(other.size_)),
        data_(std::move(other.data_)),
        grad_(std::move(other.grad_)),
        children_(std::move(other.children_)),
        grad_fn_(std::move(other.grad_fn_)),
        op_(std::move(other.op_)),
        offset_(other.offset_),
        strides_(std::move(other.strides_)) {}

  std::shared_ptr<Tensor> get_shared() { return this->shared_from_this(); }

  DataProxy data_proxy();
  GradProxy grad_proxy();

  void backward();
  void zero_grad();
  std::shared_ptr<Tensor> sum();
  std::shared_ptr<Tensor> relu();
  std::shared_ptr<Tensor> sigmoid();
  std::shared_ptr<Tensor> max();
  std::shared_ptr<Tensor> exponential();
  std::shared_ptr<Tensor> ln();
  std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> other);
  std::shared_ptr<Tensor> select(std::vector<size_t> indexes);
  std::shared_ptr<Tensor> softmax();

  std::shared_ptr<Tensor> select_data(std::vector<size_t> indexes);
  void put_data(std::vector<size_t> indexes, float value);
  std::shared_ptr<Tensor> select_grad(std::vector<size_t> indexes);
  void put_grad(std::vector<size_t> indexes, float value);

  float item();

  std::string repr() {
    // '<cudagrad.Tensor([1,], [1,]) object at 0x600001874018>'
    // '<cudagrad.Tensor([10,], [1, 1, 1, ...]) object at 0x600001874018>'

    // SIZE
    std::ostringstream oss_s;
    int i = 0;
    for (const auto &s : size_) {
      if (i >= 3) {
        break;
      }
      oss_s << s << ", ";
      ++i;
    }
    if (size_.size() >= 3) {
      oss_s << "...";
    }
    std::string s_str = oss_s.str();

    // DATA
    std::ostringstream oss_d;
    i = 0;
    for (const auto &d : data_) {
      if (i >= 3) {
        break;
      }
      oss_d << d << ", ";
      ++i;
    }
    if (data_.size() >= 3) {
      oss_d << "...";
    }
    std::string d_str = oss_d.str();

    // ADDRESS
    std::stringstream ss;
    ss << std::hex << (uintptr_t)this;
    std::string address = ss.str();

    return std::string("<Tensor([") + s_str + std::string("], [") + d_str +
           std::string("]) ") + std::string("object at 0x") + address +
           std::string(" ") + op_ + std::string(">");
  }

  void size() {
    for (auto x : size_) std::cout << x << std::endl;
  }
  void data() {
    for (auto x : data_) std::cout << x << std::endl;
  }
  void grad() {
    for (auto x : grad_) std::cout << x << std::endl;
  }
  void children() {
    for (auto x : children_) std::cout << x << std::endl;
  }

  std::vector<size_t> get_size() const { return size_; }

  std::vector<float> get_data() const { return data_; }

  std::vector<float> get_grad() const { return grad_; }

  void graph() {
    // this is to compensate for no default args pybind11
    _graph(0);
  }

  void _graph(int depth = 0) {
    auto print_if_not_leaf = [](std::string c) -> std::string {
      if (c != DEFAULT_BACKWARD_STRING) return c;
      return std::string(" ");
    };
    std::string tab(depth, ' ');
    std::string displayed_op = print_if_not_leaf(op_);
    std::cout << tab << this << " " << displayed_op << std::endl;
    for (auto c : children_) {
      c.get()->_graph(depth + 2);
    }
  }

  float &data(const std::vector<size_t> &indices) {
    // checkIndicesBounds(indices);

    size_t idx = offset_;
    for (size_t i = 0; i < indices.size(); ++i) {
      idx += indices[i] * strides_[i];
    }
    return get_shared().get()->data_[idx];
  }

  float &grad(const std::vector<size_t> &indices) {
    // checkIndicesBounds(indices);

    size_t idx = offset_;
    for (size_t i = 0; i < indices.size(); ++i) {
      idx += indices[i] * strides_[i];
    }
    return get_shared().get()->grad_[idx];
  }

  static std::shared_ptr<Tensor> ones(const std::vector<size_t> &shape) {
    std::vector<float> data(std::accumulate(shape.begin(), shape.end(), 1,
                                            std::multiplies<size_t>()),
                            1.0f);
    return tensor(shape, data);
  }

  static std::shared_ptr<Tensor> zeros(const std::vector<size_t> &shape) {
    std::vector<float> data(std::accumulate(shape.begin(), shape.end(), 1,
                                            std::multiplies<size_t>()),
                            0.0f);
    return tensor(shape, data);
  }

  static std::shared_ptr<Tensor> explode(const std::vector<size_t> &shape,
                                         float value) {
    std::vector<float> data(std::accumulate(shape.begin(), shape.end(), 1,
                                            std::multiplies<size_t>()),
                            value);
    return tensor(shape, data);
  }

  static std::shared_ptr<Tensor> rand(const std::vector<size_t> &shape) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    size_t total_size = std::accumulate(shape.begin(), shape.end(), 1,
                                        std::multiplies<size_t>());
    std::vector<float> data(total_size);
    for (size_t i = 0; i < total_size; ++i) {
      data[i] = dis(gen);
    }

    return tensor(shape, data);
  }

  static void print(std::ostream &os, const std::shared_ptr<Tensor> &tensor,
                    std::vector<size_t> indices, size_t depth,
                    bool print_grad = false) {
    if (depth == tensor.get()->size_.size() - 1) {
      os << "[";
      for (size_t i = 0; i < tensor.get()->size_[depth]; ++i) {
        indices.push_back(i);
        if (print_grad) {
          os << tensor.get()->grad(indices);
        } else {
          os << tensor.get()->data(indices);
        }
        indices.pop_back();
        if (i != tensor.get()->size_[depth] - 1) {
          os << ", ";
        }
      }
      os << "]";
    } else {
      os << "[";
      for (size_t i = 0; i < tensor.get()->size_[depth]; ++i) {
        indices.push_back(i);
        print(os, tensor, indices, depth + 1, print_grad);
        indices.pop_back();
        if (i != tensor.get()->size_[depth] - 1) {
          os << ",\n";
          os << std::string(depth + 1, ' ');
        }
      }
      os << "]";
    }
  }

  friend std::ostream &operator<<(std::ostream &os,
                                  const std::shared_ptr<Tensor> &tensor) {
    print(os, tensor, {}, 0);
    os << std::endl;
    print(os, tensor, {}, 0, true);
    return os;
  }

 private:
  void _backward();

  size_t _product(std::vector<size_t> size) {
    size_t product = 1;
    for (size_t s : size) {
      product *= s;
    }
    return product;
  }

  int _dot(std::vector<size_t> a, std::vector<size_t> b) {
    assert(a.size() == b.size());
    size_t ans = 0;
    for (size_t i = 0; i < a.size(); ++i) {
      ans += a[i] * b[i];
    }
    return ans;
  }

  void _computeStrides() {
    strides_.resize(size_.size());
    size_t stride = 1;
    for (int i = size_.size() - 1; i >= 0; i--) {
      strides_[i] = stride;
      stride *= size_[i];
    }
  }
};

// Proxys help pybind11 do __getitem__ __setitem__ on the .data/.grad member

class DataProxy {
 public:
  explicit DataProxy(Tensor &tensor) : parent_tensor(tensor) {}

  // std::vector<float> operator()() { return parent_tensor.get_data(); }

  std::shared_ptr<Tensor> get(std::vector<size_t> indexes) {
    return parent_tensor.select_data(indexes);
  }

  // std::shared_ptr<Tensor> get(const py::slice& index) {
  //   return parent_tensor.select_data(std::vector<size_t>{0, 0});
  // }

  void set(std::vector<size_t> indexes, float value) {
    parent_tensor.put_data(indexes, value);
  }

  void set(const py::slice &index, std::shared_ptr<Tensor> tensor) {
    // TODO(yrmo): only [:] supported CHECK!
    parent_tensor.data_ = tensor.get()->data_;
  }

 private:
  Tensor &parent_tensor;
};

class GradProxy {
 public:
  explicit GradProxy(Tensor &tensor) : parent_tensor(tensor) {}

  // bit hacky
  std::shared_ptr<Tensor> operator()() {
    return tensor(parent_tensor.size_, parent_tensor.grad_);
  }

  std::shared_ptr<Tensor> get(std::vector<size_t> indexes) {
    return parent_tensor.select_grad(indexes);
  }

  void set(std::vector<size_t> indexes, float value) {
    parent_tensor.put_grad(indexes, value);
  }

 private:
  Tensor &parent_tensor;
};

DataProxy Tensor::data_proxy() { return DataProxy(*this); }
GradProxy Tensor::grad_proxy() { return GradProxy(*this); }

std::vector<float> broadcast(std::vector<size_t>, std::vector<float>,
                             std::vector<size_t>);

template <typename T>
std::shared_ptr<Tensor> binaryElementwiseOperator(
    std::shared_ptr<Tensor> lhs, std::shared_ptr<Tensor> rhs,
    std::function<float(float, float)> func, std::string op,
    std::shared_ptr<T> backward) {
  std::vector<std::shared_ptr<Tensor>> children;
  children.push_back(lhs);
  children.push_back(rhs);

  if (lhs.get()->data_.size() < rhs.get()->data_.size()) {
    lhs.get()->data_ =
        broadcast(lhs.get()->size_, lhs.get()->data_, rhs.get()->size_);
  } else if (lhs.get()->data_.size() > rhs.get()->data_.size()) {
    rhs.get()->data_ =
        broadcast(rhs.get()->size_, rhs.get()->data_, lhs.get()->size_);
  }
  assert(lhs.get()->data_.size() == rhs.get()->data_.size());
  std::vector<float> result_data(lhs.get()->data_.size());
  for (size_t i = 0; i < lhs.get()->data_.size(); ++i) {
    result_data[i] = func(lhs.get()->data_[i], rhs.get()->data_[i]);
  }
  return std::make_shared<Tensor>(lhs.get()->size_, result_data, children,
                                  std::move(backward), op);
}

std::shared_ptr<Tensor> operator+(std::shared_ptr<Tensor> lhs,
                                  std::shared_ptr<Tensor> rhs) {
  return binaryElementwiseOperator<AddBackward>(
      lhs, rhs, std::plus<float>(), "AddBackward",
      std::make_shared<AddBackward>());
}

std::shared_ptr<Tensor> operator-(std::shared_ptr<Tensor> lhs,
                                  std::shared_ptr<Tensor> rhs) {
  return binaryElementwiseOperator<SubBackward>(
      lhs, rhs, std::minus<float>(), "MinusBackward",
      std::make_shared<SubBackward>());
}

std::shared_ptr<Tensor> operator*(std::shared_ptr<Tensor> lhs,
                                  std::shared_ptr<Tensor> rhs) {
  return binaryElementwiseOperator<MulBackward>(
      lhs, rhs, std::multiplies<float>(), "MulBackward",
      std::make_shared<MulBackward>());
}

std::shared_ptr<Tensor> operator/(std::shared_ptr<Tensor> lhs,
                                  std::shared_ptr<Tensor> rhs) {
  return binaryElementwiseOperator<DivBackward>(
      lhs, rhs, std::divides<float>(), "DivBackward",
      std::make_shared<DivBackward>());
}

std::shared_ptr<Tensor> operator==(std::shared_ptr<Tensor> lhs,
                                   std::shared_ptr<Tensor> rhs) {
  assert(lhs.get()->data_.size() == rhs.get()->data_.size());
  std::vector<float> equal;
  equal.reserve(lhs.get()->data_.size());
  for (size_t i = 0; i < lhs.get()->data_.size(); ++i) {
    equal.push_back(lhs.get()->data_[i] == rhs.get()->data_[i] ? 1.0f : 0.0f);
  }
  assert(lhs.get()->data_.size() == equal.size());
  return tensor(std::vector<size_t>{(lhs.get()->data_.size())}, equal);
}

std::shared_ptr<Tensor> operator!=(std::shared_ptr<Tensor> lhs,
                                   std::shared_ptr<Tensor> rhs) {
  assert(lhs.get()->data_.size() == rhs.get()->data_.size());
  std::vector<float> equal;
  equal.reserve(lhs.get()->data_.size());
  for (size_t i = 0; i < lhs.get()->data_.size(); ++i) {
    equal.push_back(lhs.get()->data_[i] == rhs.get()->data_[i] ? 0.0f : 1.0f);
  }
  assert(lhs.get()->data_.size() == equal.size());
  return tensor(std::vector<size_t>{(lhs.get()->data_.size())}, equal);
}

template <typename T>
std::shared_ptr<Tensor> binaryForwardOperator(std::shared_ptr<Tensor> lhs,
                                              std::shared_ptr<Tensor> rhs,
                                              T forward) {
  UNUSED(lhs, rhs);
  return (*forward)();  // forward.get()();
}

// std::shared_ptr<Tensor> Tensor::select(std::vector<size_t> indexes) {
//   // TODO(yrmo): nllloss needs size to be one? not sure see bc01cbf
//   // assert(indexes.size() == 1);
//   size_t idx = _dot(indexes, strides_);
//   std::vector<float> selected(1, data_[idx]);
//   auto grad_fn = std::make_shared<SelectBackward1>(idx);
//   return std::make_shared<Tensor>(
//     std::vector<size_t>{1},
//     selected,
//     std::vector<std::shared_ptr<Tensor>>{get_shared()},
//     grad_fn,
//     "SelectBackward1"
//   );
// }

std::shared_ptr<Tensor> Tensor::matmul(std::shared_ptr<Tensor> other) {
  std::shared_ptr<MatMulForward> f =
      std::make_shared<MatMulForward>(this->get_shared(), other);
  return binaryForwardOperator<std::shared_ptr<MatMulForward>>(
      this->get_shared(), other, f);
}

std::shared_ptr<Tensor> Tensor::sum() {
  std::vector<float> total(1, 0.0f);
  for (float x : data_) {
    total[0] += x;
  }
  return std::make_shared<Tensor>(
      std::vector<size_t>{1}, std::vector<float>{total},
      std::vector<std::shared_ptr<Tensor>>{get_shared()},
      std::make_shared<SumBackward>(), "SumBackward");
}

std::shared_ptr<Tensor> Tensor::max() {
  float ans = data_[0];
  for (float x : data_) {
    if (x > ans) {
      ans = x;
    }
  }
  return std::make_shared<Tensor>(
      std::vector<size_t>{1}, std::vector<float>{ans},
      std::vector<std::shared_ptr<Tensor>>{get_shared()},
      std::make_shared<MaxBackward>(), "MaxBackward");
}

std::shared_ptr<Tensor> Tensor::relu() {
  std::vector<float> result_data(data_.size());
  for (size_t i = 0; i < data_.size(); ++i) {
    result_data[i] = std::max(0.0f, data_[i]);
  }
  return std::make_shared<Tensor>(
      size_, result_data, std::vector<std::shared_ptr<Tensor>>{get_shared()},
      std::make_shared<ReluBackward>(), "ReluBackward");
}

std::shared_ptr<Tensor> Tensor::sigmoid() {
  std::vector<float> result_data(data_.size());
  for (size_t i = 0; i < data_.size(); ++i) {
    result_data[i] = 1.0f / (1.0f + exp(-data_[i]));
  }
  return std::make_shared<Tensor>(
      size_, result_data, std::vector<std::shared_ptr<Tensor>>{get_shared()},
      std::make_shared<SigmoidBackward>(), "SigmoidBackward");
}

std::shared_ptr<Tensor> Tensor::ln() {
  std::vector<float> result_data(data_.size());
  for (size_t i = 0; i < data_.size(); ++i) {
    result_data[i] = log(data_[i]);
  }
  return std::make_shared<Tensor>(
      size_, result_data, std::vector<std::shared_ptr<Tensor>>{get_shared()},
      std::make_shared<LnBackward>(), "LnBackward");
}

std::shared_ptr<Tensor> Tensor::softmax() {
    float max_value = *std::max_element(data_.begin(), data_.end());
    std::vector<float> exp_data(data_.size());
    float sum_exp = 0.0f;
    for (size_t i = 0; i < data_.size(); ++i) {
        exp_data[i] = std::exp(data_[i] - max_value);
        sum_exp += exp_data[i];
    }
    for (size_t i = 0; i < exp_data.size(); ++i) {
        exp_data[i] /= sum_exp;
    }
    return std::make_shared<Tensor>(
        size_,
        exp_data,
        std::vector<std::shared_ptr<Tensor>>{get_shared()},
        std::make_shared<SoftmaxBackward>(shared_from_this(), exp_data),
        "SoftmaxBackward"
    );
}

std::shared_ptr<Tensor> Tensor::exponential() {
  std::vector<float> result_data(data_.size());
  for (size_t i = 0; i < data_.size(); ++i) {
    result_data[i] = exp(data_[i]);
  }
  return std::make_shared<Tensor>(
      size_, result_data, std::vector<std::shared_ptr<Tensor>>{get_shared()},
      std::make_shared<ExpBackward>(), "ExpBackward");
}

std::shared_ptr<Tensor> tensor(std::initializer_list<size_t> size,
                               std::initializer_list<float> data) {
  return std::make_shared<Tensor>(size, data);
}

std::shared_ptr<Tensor> tensor(std::vector<size_t> size,
                               std::vector<float> data) {
  return std::make_shared<Tensor>(size, data);
}

std::shared_ptr<Tensor> tensor(
    std::vector<size_t> size, std::vector<float> data,
    std::vector<std::shared_ptr<Tensor>> children = {},
    std::shared_ptr<AutoGradBackward> grad_fn =
        std::make_shared<AutoGradBackward>(),
    std::string op = DEFAULT_BACKWARD_STRING) {
  return std::make_shared<Tensor>(size, data, children, std::move(grad_fn), op);
}

/*
std::shared_ptr<Tensor> Tensor::sum() {
  std::vector<float> total(1, 0.0f);
  for (float x : data_) {
    total[0] += x;
  }
  return std::make_shared<Tensor>(
      std::vector<int>{1}, std::vector<float>{total},
      std::vector<std::shared_ptr<Tensor>>{get_shared()},
      std::make_shared<SumBackward>(), 's');
}
*/

// TODO(yrmo): DRY tensor access

std::shared_ptr<Tensor> Tensor::select_data(std::vector<size_t> indexes) {
  size_t product = 0;
  if (std::accumulate(indexes.begin(), indexes.end(), 0) > 0) {
    product = _dot(indexes, strides_);
  }
  return std::make_shared<Tensor>(
      std::vector<size_t>{1}, std::vector<float>{data_[product]},
      std::vector<std::shared_ptr<Tensor>>{get_shared()},
      std::make_shared<SelectBackward>(), "SelectBackward");
}

std::shared_ptr<Tensor> Tensor::select_grad(std::vector<size_t> indexes) {
  size_t product = 0;
  if (std::accumulate(indexes.begin(), indexes.end(), 0) > 0) {
    product = _dot(indexes, strides_);
  }
  return std::make_shared<Tensor>(
      std::vector<size_t>{1}, std::vector<float>{grad_[product]},
      std::vector<std::shared_ptr<Tensor>>{get_shared()},
      std::make_shared<SelectBackward>(), "SelectGradBackward");
}

// TODO(yrmo): this is untracked, does pytorch track assignment?

void Tensor::put_data(std::vector<size_t> indexes, float value) {
  size_t product = 0;
  if (std::accumulate(indexes.begin(), indexes.end(), 0) > 0) {
    product = _dot(indexes, strides_);
  }
  data_[product] = value;
}

void Tensor::put_grad(std::vector<size_t> indexes, float value) {
  size_t product = 0;
  if (std::accumulate(indexes.begin(), indexes.end(), 0) > 0) {
    product = _dot(indexes, strides_);
  }
  grad_[product] = value;
}

void debug_inputs(std::shared_ptr<Tensor> grad_output,
                  std::vector<std::shared_ptr<Tensor>> grad_inputs,
                  std::string operation) {
  UNUSED(grad_output, grad_inputs, operation);
  // std::cout << std::string("--------------------") << std::string("INPUT") <<
  // std::string("--------------------") << std::endl; std::cout <<
  // std::string("--------------------") << operation <<
  // std::string("--------------------") << std::endl; std::cout << grad_output
  // << std::endl; for (auto grad_input : grad_inputs) {
  //   std::cout << grad_input << std::endl;
  // }
}

void debug_outputs(std::shared_ptr<Tensor> grad_output,
                   std::vector<std::shared_ptr<Tensor>> grad_inputs,
                   std::string operation) {
  UNUSED(grad_output, grad_inputs, operation);
  // std::cout << std::string("--------------------") << std::string("OUTPUT")
  // << std::string("--------------------") << std::endl; std::cout <<
  // std::string("--------------------") << operation <<
  // std::string("--------------------") << std::endl; std::cout << grad_output
  // << std::endl; for (auto grad_input : grad_inputs) {
  //   std::cout << grad_input << std::endl;
  // }
}

struct AutoGradBackward {
  AutoGradBackward() = default;
  virtual ~AutoGradBackward() = default;

  virtual void apply(std::shared_ptr<Tensor> grad_output,
                     std::vector<std::shared_ptr<Tensor>> grad_inputs) {
    UNUSED(grad_output, grad_inputs);
    throw std::runtime_error("apply() is not implemented");
  }
};

std::vector<float> broadcast(std::vector<size_t> from, std::vector<float> data,
                             std::vector<size_t> to) {

  // for (auto x : from) {
  //   std::cout << x << std::endl;
  // }
  // std::cout << std::string("--------------------------------") << std::endl;
  // for (auto x : to) {
  //   std::cout << x << std::endl;
  // }
  // TODO(yrmo) only scalar, vector, and matrix until a nn needs rank > 2
  assert(from.size() < 3);
  assert(to.size() < 3);

  // 1D (scalar) -> 1D
  // e.g. {1} -> {3}
  if (from.size() == 1 && to.size() == 1 && from[0] == 1) {
    return std::vector<float>(to[0], data[0]);
  }
  // 1D -> 1D (no broadcast)
  // e.g. {2} -> {2}
  if (from.size() == 1 && to.size() == 1 && from[0] == to[0]) {
    return std::vector<float>(data.begin(), data.end());
  }
  // 1D (scalar) -> 2D
  // e.g. {1} -> {2, 2}
  if (from.size() == 1 && to.size() == 2 && from[0] == 1) {
    return std::vector<float>(to[0] * to[1], data[0]);
  }
  // 1D -> 1D
  // e.g. {4} -> {1}
  if (from.size() == 1 && to.size() == 1 && to[0] == 1) {
    return std::vector<float>(to[0], data[0]);
  }
  // MNIST TEMP HARD CODE
  if (from.size() == 2 && to.size() == 1 && from[0] == 1 && from[1] == 10 && to[0] == 1) {
    return std::vector<float>(to[0], data[0]);
  }

  // 2D -> 1D
  // e.g. {1, 2} -> {2}
  if (from.size() == 2 && to.size() == 1 && from[0] == 1) {
    assert(from[1] == to[0]);
    return std::vector<float>(data.begin(), data.end());
  }
  // 2D -> 1D
  // e.g. {2, 1} -> {2}
  if (from.size() == 2 && to.size() == 1 && from[1] == 1) {
    assert(from[0] == to[0]);
    return std::vector<float>(data.begin(), data.end());
  }

  // 2D -> 2D
  // e.g. {1, m} -> {m, m}
  if (from.size() == 2 && to.size() == 2 && from[0] == 1) {
    std::vector<float> result;
    result.reserve(to[0] * to[1]);
    for (size_t i = 0; i < to[0]; ++i) {
      result.insert(result.end(), data.begin(), data.end());
    }
    return result;
  } else if (from.size() == 2 && to.size() == 2 && from[1] == 1) {
    // 2D -> 2D
    // e.g. {n, 1} -> {n, n}
    std::vector<float> result;
    result.reserve(to[0] * to[1]);
    for (size_t i = 0; i < to[0]; ++i) {
      for (size_t j = 0; j < to[1]; ++j) {
        result.push_back(data[i]);
      }
    }
    return result;
  } else if (from.size() == 2 && to.size() == 2 && to[0] == from[0] &&
             to[1] == from[1]) {
    // 2D -> 2D (NOOP)
    // e.g. {n, m} -> {n, m}
    return data;
  }

  std::cout << "from" << " " << std::endl;
  for (auto x: from) {
    std::cout << x << " " << std::endl;
  }
  std::cout << "to" << " " << std::endl;
  for (auto x: to) {
    std::cout << x << " " << std::endl;
  }
  throw std::runtime_error("Invalid broadcast");
}

struct SoftmaxBackward : public AutoGradBackward {
    std::shared_ptr<Tensor> input_;
    std::vector<float> output_data_;

    SoftmaxBackward(std::shared_ptr<Tensor> input, const std::vector<float>& output_data)
        : input_(std::move(input)), output_data_(output_data) {}

    void apply(std::shared_ptr<Tensor> grad_output,
               std::vector<std::shared_ptr<Tensor>> grad_inputs) override {
        assert(grad_inputs.size() == 1);
        std::shared_ptr<Tensor> input = grad_inputs[0];

        for (size_t i = 0; i < input->grad_.size(); ++i) {
            float grad_sum = 0.0f;
            for (size_t j = 0; j < output_data_.size(); ++j) {
                if (i == j) {
                    grad_sum += grad_output->grad_[j] * output_data_[i] * (1.0f - output_data_[j]);
                } else {
                    grad_sum -= grad_output->grad_[j] * output_data_[i] * output_data_[j];
                }
            }
            input->grad_[i] += grad_sum;
        }
    }
};

struct SelectBackward1 : public AutoGradBackward {
  size_t index_;

  SelectBackward1(size_t idx) : index_(idx) {}

    void apply(std::shared_ptr<Tensor> grad_output,
               std::vector<std::shared_ptr<Tensor>> grad_inputs) override {
        debug_inputs(grad_output, grad_inputs, "SelectBackward1");
        assert(grad_inputs.size() == 1);
        std::shared_ptr<Tensor> input = grad_inputs[0];

        input->grad_[index_] += grad_output->grad_[0];
        
        debug_outputs(grad_output, grad_inputs, "SelectBackward1");
    }
};

struct AddBackward : public AutoGradBackward {
  AddBackward() = default;

  void apply(std::shared_ptr<Tensor> grad_output,
             std::vector<std::shared_ptr<Tensor>> grad_inputs) override {
    debug_inputs(grad_output, grad_inputs, "AddBackward");
    for (std::shared_ptr<Tensor> grad_input : grad_inputs) {
      std::vector<float> broadcast_output_grad =
          broadcast(grad_output.get()->size_, grad_output.get()->grad_,
                    grad_input.get()->size_);
      for (size_t i = 0; i < grad_input.get()->grad_.size(); ++i) {
        grad_input.get()->grad_[i] +=
            broadcast_output_grad[i];  // grad_output.get()->grad_[i];
      }
    }
    debug_outputs(grad_output, grad_inputs, "AddBackward");
  }
};

struct MaxBackward : public AutoGradBackward {
  MaxBackward() = default;

  void apply(std::shared_ptr<Tensor> grad_output,
             std::vector<std::shared_ptr<Tensor>> grad_inputs) override {
    for (std::shared_ptr<Tensor> grad_input : grad_inputs) {
      int index = 0;
      float biggest = grad_input.get()->data_[0];
      for (int i = 0; i < grad_input.get()->data_.size(); i++) {
        if (grad_input.get()->data_[i] > biggest) {
          biggest = grad_input.get()->data_[i];
          index = i;
        }
      }
      grad_input.get()->grad_[index] += grad_output.get()->grad_[0];
    }
  }
};

struct SubBackward : public AutoGradBackward {
  SubBackward() = default;

  void apply(std::shared_ptr<Tensor> grad_output,
             std::vector<std::shared_ptr<Tensor>> grad_inputs) override {
    for (size_t i = 0; i < grad_inputs[0].get()->grad_.size(); ++i) {
      grad_inputs[0].get()->grad_[i] += grad_output.get()->grad_[0];
    }
    for (size_t i = 0; i < grad_inputs[1].get()->grad_.size(); ++i) {
      grad_inputs[1].get()->grad_[i] -= grad_output.get()->grad_[0];
    }
  }
};

struct MulBackward : public AutoGradBackward {
  MulBackward() = default;

  void apply(std::shared_ptr<Tensor> grad_output,
             std::vector<std::shared_ptr<Tensor>> grad_inputs) override {
    for (size_t i = 0; i < grad_inputs[0].get()->grad_.size(); ++i) {
      grad_inputs[0].get()->grad_[i] +=
          grad_output.get()->grad_[0] * grad_inputs[1].get()->data_[i];
    }
    for (size_t i = 0; i < grad_inputs[1].get()->grad_.size(); ++i) {
      grad_inputs[1].get()->grad_[i] +=
          grad_output.get()->grad_[0] * grad_inputs[0].get()->data_[i];
    }
  }
};

struct DivBackward : public AutoGradBackward {
  DivBackward() = default;

  void apply(std::shared_ptr<Tensor> grad_output,
             std::vector<std::shared_ptr<Tensor>> grad_inputs) override {
    for (size_t i = 0; i < grad_inputs[0].get()->grad_.size(); ++i) {
      // pro tip: don't put 1 instead of 1.0f!
      float update =
          grad_output.get()->grad_[i] * (1.0f / grad_inputs[1].get()->data_[i]);
      grad_inputs[0].get()->grad_[i] += update;
    }
    for (size_t i = 0; i < grad_inputs[1].get()->grad_.size(); ++i) {
      float update =
          grad_output.get()->grad_[i] *
          -(grad_inputs[0].get()->data_[i] /
            (grad_inputs[1].get()->data_[i] * grad_inputs[1].get()->data_[i]));
      grad_inputs[1].get()->grad_[i] += update;
    }
  }
};

struct ReluBackward : public AutoGradBackward {
  ReluBackward() = default;

  void apply(std::shared_ptr<Tensor> grad_output,
             std::vector<std::shared_ptr<Tensor>> grad_inputs) override {
    assert(grad_inputs.size() == 1);
    std::shared_ptr<Tensor> input = grad_inputs[0];
    for (size_t i = 0; i < input.get()->grad_.size(); ++i) {
      input.get()->grad_[i] +=
          grad_output.get()->grad_[i] * (input.get()->data_[i] > 0 ? 1 : 0);
    }
  }
};

struct SigmoidBackward : public AutoGradBackward {
  SigmoidBackward() = default;

  void apply(std::shared_ptr<Tensor> grad_output,
             std::vector<std::shared_ptr<Tensor>> grad_inputs) override {
    debug_inputs(grad_output, grad_inputs, "SigmoidBackward");
    assert(grad_inputs.size() == 1);
    std::shared_ptr<Tensor> input = grad_inputs[0];
    for (size_t i = 0; i < input.get()->grad_.size(); ++i) {
      auto s = 1.0f / (1.0f + exp(-input.get()->data_[i]));
      auto temp_debug = grad_output.get()->grad_[i] * ((s) * (1 - s));
      input.get()->grad_[i] += temp_debug;
    }
    debug_outputs(grad_output, grad_inputs, "SigmoidBackward");
  }
};

struct ExpBackward : public AutoGradBackward {
  ExpBackward() = default;

  void apply(std::shared_ptr<Tensor> grad_output,
             std::vector<std::shared_ptr<Tensor>> grad_inputs) override {
    assert(grad_inputs.size() == 1);
    std::shared_ptr<Tensor> input = grad_inputs[0];
    for (size_t i = 0; i < input.get()->grad_.size(); ++i) {
      input.get()->grad_[i] +=
          grad_output.get()->grad_[i] * exp(input.get()->data_[i]);
    }
  }
};

struct LnBackward : public AutoGradBackward {
  LnBackward() = default;

  void apply(std::shared_ptr<Tensor> grad_output,
             std::vector<std::shared_ptr<Tensor>> grad_inputs) override {
    assert(grad_inputs.size() == 1);
    std::shared_ptr<Tensor> input = grad_inputs[0];
    for (size_t i = 0; i < input.get()->grad_.size(); ++i) {
      input.get()->grad_[i] +=
          grad_output.get()->grad_[i] * (1.0f / input.get()->data_[i]);
    }
  }
};

struct AutoGradForward {
  AutoGradForward() = default;
  virtual ~AutoGradForward() = default;

  virtual std::shared_ptr<Tensor> operator()()
      // (const std::shared_ptr<Tensor>& lhs, const std::shared_ptr<Tensor>&
      // rhs)
      = 0;
};

std::vector<float> _matmul(const std::vector<size_t> &lhs_size,
                           const std::vector<float> &lhs_data,
                           const std::vector<size_t> &rhs_size,
                           const std::vector<float> &rhs_data) {
  size_t m = lhs_size[0];
  size_t n = rhs_size[1];
  size_t k = lhs_size[1];

  std::vector<float> result_data(m * n, 0.0f);

  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      for (size_t p = 0; p < k; ++p) {
        result_data[i * n + j] += lhs_data[i * k + p] * rhs_data[p * n + j];
      }
    }
  }

  return result_data;
}

struct MatMulForward : public AutoGradForward {
  std::shared_ptr<Tensor> lhs_;
  std::shared_ptr<Tensor> rhs_;

  MatMulForward(const std::shared_ptr<Tensor> &lhs,
                const std::shared_ptr<Tensor> &rhs)
      : lhs_(lhs), rhs_(rhs) {
    if (lhs.get()->size_.size() != 2 || rhs.get()->size_.size() != 2 ||
        lhs.get()->size_[1] != rhs.get()->size_[0]) {
      throw std::runtime_error("Invalid dimensions for matrix multiplication.");
    }
  }

  std::shared_ptr<Tensor> operator()() {
    std::vector<float> result_data =
        _matmul(lhs_.get()->size_, lhs_.get()->data_, rhs_.get()->size_,
                rhs_.get()->data_);

    std::vector<size_t> result_size = {lhs_.get()->size_[0],
                                       rhs_.get()->size_[1]};
    std::vector<std::shared_ptr<Tensor>> result_children = {lhs_, rhs_};
    std::shared_ptr<MatMulBackward> result_grad_fn =
        std::make_shared<MatMulBackward>();

    return std::make_shared<Tensor>(result_size, result_data,
                                    std::move(result_children),
                                    std::move(result_grad_fn),
                                    // TODO(yrmo) Is this MatMulForward?
                                    "MatMulBackward");
  }
};

std::vector<float> transpose(const std::vector<size_t> &size,
                             const std::vector<float> &data) {
  size_t rows = size[0];
  size_t cols = size[1];
  std::vector<float> transposed_data(cols * rows, 0.0f);

  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      transposed_data[j * rows + i] = data[i * cols + j];
    }
  }

  return transposed_data;
}

struct MatMulBackward : public AutoGradBackward {
  MatMulBackward() = default;

  void apply(std::shared_ptr<Tensor> grad_output,
             std::vector<std::shared_ptr<Tensor>> grad_inputs) override {
    debug_inputs(grad_output, grad_inputs, "MatMulBackward");
    // std::cout << std::string("matmulbackward") << std::endl;
    // for (std::shared_ptr<Tensor> grad_input : grad_inputs) {
    //   std::cout << std::string("input") << std::endl;
    //   for (size_t i = 0; i < grad_output.get()->grad_.size(); ++i) {
    //     // std::cout << grad_input.get()->grad_[i] << std::endl;
    //     std::cout << grad_output.get()->grad_[i] << std::endl;
    //   }
    // }

    auto a = grad_inputs[0];
    auto b = grad_inputs[1];

    auto b_transposed_size =
        std::vector<size_t>{b.get()->size_[1], b.get()->size_[0]};
    auto b_transposed_data = transpose(b.get()->size_, b.get()->data_);

    auto a_transposed_size =
        std::vector<size_t>{a.get()->size_[1], a.get()->size_[0]};
    auto a_transposed_data = transpose(a.get()->size_, a.get()->data_);

    auto grad_a_data =
        _matmul(grad_output.get()->size_, grad_output.get()->grad_,
                b_transposed_size, b_transposed_data);
    auto grad_b_data =
        _matmul(a_transposed_size, a_transposed_data, grad_output.get()->size_,
                grad_output.get()->grad_);

    for (size_t i = 0; i < a.get()->grad_.size(); ++i) {
      a.get()->grad_[i] += grad_a_data[i];
    }

    for (size_t i = 0; i < b.get()->grad_.size(); ++i) {
      b.get()->grad_[i] += grad_b_data[i];
    }
    debug_outputs(grad_output, grad_inputs, "MatMulBackward");
  }
};

void Tensor::_backward() {
  grad_fn_.get()->apply(get_shared(), children_);
  for (auto child : children_) {
    if (child.get()->grad_fn_ != nullptr) {
      child.get()->_backward();
    }
  }
}

void Tensor::backward() {
  assert(data_.size() == 1);
  grad_ = std::vector<float>(_product(size_), 1.0f);
  grad_fn_.get()->apply(get_shared(), children_);
  for (auto child : children_) {
    if (child.get()->grad_fn_ != nullptr) {
      assert(child.get()->op_ != DEFAULT_BACKWARD_STRING);
      child.get()->_backward();
    }
  }
}

float Tensor::item() {
  assert(data_.size() == 1);
  return data_[0];
}

void Tensor::zero_grad() {
  grad_ = std::vector<float>(_product(size_), 0.0f);
  for (auto child : children_) {
    if (child.get()->children_.size() != 0) {
      child.get()->zero_grad();
    }
  }
}

// namespace nn {

// // TODO(yrmo): nonlinearity of Neuron need to pass and drill down from MLP
// //              static_cast<bool>(vec_.size() - 1) something liek this for
// MLP

// // keep it simple to start
// class Neuron {
//  public:
//   std::vector<std::shared_ptr<Tensor>> weights_;
//   std::shared_ptr<Tensor> bias_;
//   float rate_;

//   Neuron(int nin, float rate) : rate_(rate) {
//     for (size_t i = 0; i < nin; ++i) {
//       weights_.push_back(Tensor::rand({1}));
//     }
//     assert(weights_.size() == nin);
//     bias_ = Tensor::rand({1});
//   }

//   std::shared_ptr<Tensor> operator()(
//       std::vector<std::shared_ptr<Tensor>> inputs) {
//     if (inputs.size() != weights_.size()) {
//       throw std::runtime_error("Neuron inputs length weights length
//       mismatch!");
//     }
//     std::shared_ptr<Tensor> ans = nullptr;
//     for (size_t i = 0; i < weights_.size(); ++i) {
//       if (ans == nullptr) {
//         ans = (weights_[i] * inputs[i]);
//       } else {
//         ans = ans + (weights_[i] * inputs[i]);
//       }
//     }
//     ans = ans + bias_;
//     return ans.get()->relu();
//   }

//   void train() {
//     for (size_t i = 0; i < weights_.size(); ++i) {
//       // std::cout << "weight before: " << weights_[i].get()->data_[0] <<
//       // std::endl;
//       weights_[i].get()->data_[0] = weights_[i].get()->data_[0] +
//                                     ((-rate_) * weights_[i].get()->grad_[0]);
//       // std::cout << "weight after: " << weights_[i].get()->data_[0] <<
//       // std::endl;
//     }
//     // std::cout << "bias before: " << bias_.get()->data_[0] << std::endl;
//     bias_.get()->data_[0] =
//         bias_.get()->data_[0] + ((-rate_) * bias_.get()->grad_[0]);
//     // std::cout << "bias after: " << bias_.get()->data_[0] << std::endl;
//   }
//   // TODO(yrmo) parameters, how do i make this like std iterator?
// };

// }  // namespace nn

}  // namespace cg

#endif  // SRC_TENSOR_HPP_
