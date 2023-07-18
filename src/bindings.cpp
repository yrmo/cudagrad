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

    m.def("hello", &cg::hello, R"pbdoc(
        Can has CUDA?

        Hello!
    )pbdoc");

    m.def("tensor", [](std::vector<int> sizes, std::vector<float> values) {
        // cast the function pointer to resolve the ambiguity
        auto func = static_cast<std::shared_ptr<cg::Tensor> (*)(std::vector<int>, std::vector<float>)>(&cg::tensor);
        return func(sizes, values);
    }, R"pbdoc(Magic tensor)pbdoc",
    py::arg("sizes"), py::arg("values"));

    py::class_<cg::Tensor, std::shared_ptr<cg::Tensor>>(m, "Tensor")
        .def(py::init<std::vector<int>, std::vector<float>>())
        .def("get_shared", &cg::Tensor::get_shared)
        .def("backward", &cg::Tensor::backward)
        .def("zero_grad", &cg::Tensor::zero_grad)
        .def("sum", &cg::Tensor::sum)
        .def("matmul", &cg::matmul)
        .def_property_readonly("size", &cg::Tensor::get_size)
        .def_property_readonly("data", &cg::Tensor::get_data)
        .def_property_readonly("grad", &cg::Tensor::get_grad)
        .def("graph", &cg::Tensor::graph)
        .def_static("ones", &cg::Tensor::ones)
        .def_static("zeros", &cg::Tensor::zeros)
        .def_static("explode", &cg::Tensor::explode)
        .def_static("rand", &cg::Tensor::rand)
        .def("__add__", [](std::shared_ptr<cg::Tensor> a, std::shared_ptr<cg::Tensor> b) { return a + b; })
        .def("__sub__", [](std::shared_ptr<cg::Tensor> a, std::shared_ptr<cg::Tensor> b) { return a - b; })
        .def("__mul__", [](std::shared_ptr<cg::Tensor> a, std::shared_ptr<cg::Tensor> b) { return a * b; })
        .def("__truediv__", [](std::shared_ptr<cg::Tensor> a, std::shared_ptr<cg::Tensor> b) { return a / b; })
        .def("__matmul__", [](std::shared_ptr<cg::Tensor> a, std::shared_ptr<cg::Tensor> b) { return cg::matmul(a, b); });

    // Add the stream output operator if you want to print your Tensor object in Python.
    // py::class_<std::ostream>(m, "ostream")
    //     .def(py::self_ns::str(py::self_ns::self))
    //     .def(py::self_ns::repr(py::self_ns::self));

    // m.def("__lshift__", [](std::ostream &os, const std::shared_ptr<cg::Tensor> &t) -> std::ostream& {
    //     os << t; return os;
    // }, py::is_operator());

    // m.def("tensor", [](std::vector<int> size, std::vector<float> data) {
    //     return cg::tensor(size, data);
    // }, R"pbdoc(Magic tensor)pbdoc", py::arg("sizes"), py::arg("values"));


#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
