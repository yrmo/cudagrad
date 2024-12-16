// Copyright 2023-2024 Ryan Moore

#include <stdio.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#ifdef _MSC_VER
#define WEAK
#else
#define WEAK __attribute__((weak))
#endif

namespace cg {

extern "C" bool cuda_available() {
#ifdef _WIN32
  HMODULE handle = LoadLibrary("nvcuda.dll");
  if (handle) {
    FreeLibrary(handle);
    return true;
  } else {
    return false;
  }
#else
  void* handle = dlopen("libcuda.so", RTLD_LAZY);
  if (handle) {
    dlclose(handle);
    return true;
  } else {
    return false;
  }
#endif
}

const char* helloFromCPU() { return "Hello, CPU!"; }

extern "C" const char* helloCPU() { return helloFromCPU(); }
extern "C" const char *helloGPU() WEAK;

}  // namespace cg
