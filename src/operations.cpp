// Copyright 2023-2024 Ryan Moore

#include <stdio.h>

namespace cg {

const char* helloFromCPU() { return "Hello, CPU!"; }

extern "C" const char* hello() { return helloFromCPU(); }

}  // namespace cg
