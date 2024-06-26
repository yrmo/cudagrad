// Copyright 2023-2024 Ryan Moore

#include <stdio.h>

#include <cassert>
#include <string>

__global__ void foo() {}

int main() {
  foo<<<1, 1>>>();
  assert(std::string(cudaGetErrorString(cudaGetLastError())) == "no error");
}
