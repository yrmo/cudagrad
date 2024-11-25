// Copyright 2023-2024 Ryan Moore
//
// Calling code “clever” is usually an insult in software engineering, since
// it means the code’s functionality is sufficiently obscure it’ll be hard to
// maintain. One exception is CUDA kernels, where squeezing out a bit more
// performance is often worth some brittleness in exchange.
//
// Greg Brockman

#include <stdio.h>

#include <cstdio>

#include "cub/cub.cuh"
#include "cuda/std/atomic"
#include "thrust/device_vector.h"

namespace cg {

__global__ void helloFromGPU(char* device_message) {
  const char* message = "Hello, GPU!";
  for (int i = 0; i < 11; ++i) {
      device_message[i] = message[i];
  }
}

extern "C" const char * hello() {
  static char host_message[11];
  char* device_message;

  cudaMalloc(&device_message, 11 * sizeof(char));

  helloFromGPU<<<1, 1>>>(device_message);
  cudaDeviceSynchronize();

  cudaMemcpy(host_message, device_message, 11 * sizeof(char), cudaMemcpyDeviceToHost);

  cudaFree(device_message);
  return host_message;
}

}  // namespace cg
