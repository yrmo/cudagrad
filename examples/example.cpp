// c++ -std=c++11 -I../src example.cpp && ./a.out
#include "cudagrad.hpp"
int main() {
  auto a = cg::tensor({2, 2}, {2.0, 3.0, 4.0, 5.0});
  auto b = cg::tensor({2, 2}, {6.0, 7.0, 8.0, 9.0});
  auto c = cg::tensor({2, 2}, {10.0, 10.0, 10.0, 10.0});
  auto d = cg::tensor({2, 2}, {11.0, 11.0, 11.0, 11.0});
  auto e = (cg::matmul(a, b) + c) * d;
  auto f = cg::sum(e);
  f.get()->backward();

  using namespace std; // NOLINT(build/namespaces)
  for (auto& x : f.get()->data_) { cout << x << endl;} // 2794
  for (auto& x : f.get()->size_) { cout << x << endl;} // 1
  for (auto& x : a.get()->grad_) { cout << x << endl; } // 143 187 143 187
  for (auto& x : b.get()->grad_) { cout << x << endl; } // 66 66 88 88
}
