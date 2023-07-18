// c++ -std=c++11 -I./src examples/xor.cpp && ./a.out

#include "cudagrad.hpp"
#include <cmath>

int main() {
  std::vector<float> a = {0.0f, 0.0f, 1.0f, 1.0f};
  std::vector<float> b = {0.0f, 1.0f, 0.0f, 1.0f};
  std::vector<float> ans = {0.0f, 1.0f, 1.0f, 0.0f}; // XOR function

  auto mlp = cg::nn::MLP(2, {2, 2, 1}, 0.025); // Lower the learning rate
  for (int epoch = 0; epoch < 301; ++epoch) {
    auto sum = cg::tensor({1}, {0.0f});
    for (int i = 0; i < a.size(); ++i) {
      std::vector<std::shared_ptr<cg::Tensor>> inputs = {
        cg::tensor({1}, {a[i]}), cg::tensor({1}, {b[i]})
      };

      auto response = mlp(inputs);
      sum = sum + ((cg::tensor({1}, {ans[i]}) - response[0]) * (cg::tensor({1}, {ans[i]}) - response[0]));
    }
    auto loss = sum / cg::tensor({1}, {static_cast<float>(a.size())});

    loss.get()->zero_grad();
    loss.get()->backward();
    mlp.train();

    // after each epoch, evaluate and print the outputs
    if (epoch % 50 == 0) {
      std::cout << "EPOCH: " << epoch << " LOSS: " << loss.get()->data_[0] << std::endl;

      for (int i = 0; i < a.size(); ++i) {
        std::vector<std::shared_ptr<cg::Tensor>> inputs = {
          cg::tensor({1}, {a[i]}), cg::tensor({1}, {b[i]})
        };
        auto response = mlp(inputs);
        if (round(response[0].get()->data_[0]) == ans[i]) {
          std::cout << a[i] << " XOR " << b[i] << " = " << "🔥" << response[0].get()->data_[0] << std::endl;
        } else {
          std::cout << a[i] << " XOR " << b[i] << " = " << "  " << response[0].get()->data_[0] << std::endl;
        }
      }
    }
  }
}
