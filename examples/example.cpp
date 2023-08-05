// c++ -std=c++11 -I./src examples/example.cpp && ./a.out
#include "tensor.hpp"
int main() {
  auto a = cg::tensor({2, 2}, {2.0, 3.0, 4.0, 5.0});
  auto b = cg::tensor({2, 2}, {6.0, 7.0, 8.0, 9.0});
  auto c = cg::tensor({2, 2}, {10.0, 10.0, 10.0, 10.0});
  auto d = cg::tensor({2, 2}, {11.0, 11.0, 11.0, 11.0});
  auto e = (a.get()->matmul(b) + c) * d;
  auto f = e.get()->sum();
  f.get()->backward();

  using namespace std;
  for (auto& x : f.get()->data_) {cout<<x<<" ";} // 2794
  for (auto& x : f.get()->size_) {cout<<x<<" ";} // 1
  for (auto& x : a.get()->grad_) {cout<<x<<" ";} // 143 187 143 187
  for (auto& x : b.get()->grad_) {cout<<x<<" ";} // 66 66 88 88
}
