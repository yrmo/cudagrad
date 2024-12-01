// Copyright 2023-2024 Ryan Moore

#include <stdio.h>
#include <dlfcn.h>

namespace cg {

extern "C" bool cuda_available() {
  void* handle = dlopen("libcuda.so", RTLD_LAZY);
  if (handle) {
    dlclose(handle);
    return true;
  } else {
    return false;
  }
}

const char* helloFromCPU() { return "Hello, CPU!"; }

extern "C" const char* helloCPU() { return helloFromCPU(); }

__attribute__((weak)) extern "C" const char* helloGPU() {}

}  // namespace cg
