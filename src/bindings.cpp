#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tensor.hpp"  // NOLINT(build/include_subdir)

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

PYBIND11_MODULE(tensor, m) {
  m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: cudagrad

        .. autosummary::
           :toctree: _generate

        hello
        tensor
        Tensor
    )pbdoc";

  m.def("hello", &cg::hello, R"pbdoc(
        Can has CUDA?

        Hello!
    )pbdoc");

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
