#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vecAdd(float* A, float* B, float* C) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  C[i] = A[i] + B[i];
}

__global__ void vecSub(float* A, float* B, float* C) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  C[i] = A[i] - B[i];
}

__global__ void vecMul(float* A, float* B, float* C) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  C[i] = A[i] * B[i];
}

__global__ void vecDiv(float* A, float* B, float* C) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  C[i] = A[i] / B[i];
}

struct Add {
    __device__ float operator()(const float& a, const float& b) const {
        return a + b;
    }
};

struct Sub {
    __device__ float operator()(const float& a, const float& b) const {
        return a - b;
    }
};

struct Mul {
    __device__ float operator()(const float& a, const float& b) const {
        return a * b;
    }
};

struct Div {
    __device__ float operator()(const float& a, const float& b) const {
        return a / b;
    }
};

template <typename T>
__global__ void vecOp(T op, float* A, float* B, float* C) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    C[i] = op(A[i], B[i]);
}

int main(void) {
  int N = 1024;
  size_t size = N * sizeof(float);

  float* h_A = (float*)malloc(size);
  float* h_B = (float*)malloc(size);
  float* h_C = (float*)malloc(size);

  for(int i = 0; i < N; ++i) {
    h_A[i] = i;
    h_B[i] = i;
  }

  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, size);
  cudaMalloc(&d_B, size);
  cudaMalloc(&d_C, size);

  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  vecOp<<<blocksPerGrid, threadsPerBlock>>>(Add(), d_A, d_B, d_C);

  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

  for(int i = 0; i < N; ++i) {
    if (h_A[i] + h_B[i] != h_C[i]) {
      printf("Results do not match at index %d!\n", i);
      break;
    }
  }

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  free(h_A);
  free(h_B);
  free(h_C);

  return 0;
}
