#include <stdio.h>

namespace cg {

void helloFromCPU() { printf("Hello, CPU!\n"); }

extern "C" void hello() {
  helloFromCPU();
}

}  // namespace cg
