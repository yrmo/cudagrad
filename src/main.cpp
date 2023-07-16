#include "cudagrad.hpp"

#include <pybind11/stl.h>
#include <pybind11/pybind11.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

int add(int i, int j) {
    return i + j;
}

namespace py = pybind11;

PYBIND11_MODULE(cudagrad, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: cudagrad

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

    m.def("add", &add, R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");

    m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers

        Some other explanation about the subtract function.
    )pbdoc");

    m.def("foo", &cg::foo, R"pbdoc(
        Add one

        Why is this here?
    )pbdoc");

    m.def("tensor", [](std::vector<int> sizes, std::vector<float> values) {
        // cast the function pointer to resolve the ambiguity
        auto func = static_cast<std::shared_ptr<cg::Tensor> (*)(std::vector<int>, std::vector<float>)>(&cg::tensor);
        return func(sizes, values);
    }, R"pbdoc(Magic tensor)pbdoc",
    py::arg("sizes"), py::arg("values"));


    // m.def("tensor", [](std::vector<int> size, std::vector<float> data) {
    //     return cg::tensor(size, data);
    // }, R"pbdoc(Magic tensor)pbdoc", py::arg("sizes"), py::arg("values"));


#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
