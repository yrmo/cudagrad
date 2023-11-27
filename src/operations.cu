// Copyright 2023 Ryan Moore
//
// Calling code “clever” is usually an insult in software engineering, since
// it means the code’s functionality is sufficiently obscure it’ll be hard to
// maintain. One exception is CUDA kernels, where squeezing out a bit more
// performance is often worth some brittleness in exchange.
//
// Greg Brockman

#include <stdio.h>

__global__ void helloFromGPU() { printf("Hello, GPU!\n"); }

extern "C" void hello() {
  helloFromGPU<<<1, 1>>>();
  cudaDeviceSynchronize();
}
