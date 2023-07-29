// Copyright 2023 Ryan Moore
//
// 'Generic' is the enemy of 'Efficient'
//
// Tim Zaman

#include <stdio.h>

__global__ void helloFromGPU() { printf("Hello, GPU!\n"); }

extern "C" void hello() {
  helloFromGPU<<<1, 1>>>();
  cudaDeviceSynchronize();
}
