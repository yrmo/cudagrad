// Copyright 2023 Ryan Moore
//
// If the implementation is easy to explain, it may be a good idea.
//
// Tim Peters

#include <sstream>
#include "tensor.hpp"

#include <pybind11/stl.h>
#include <pybind11/pybind11.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

PYBIND11_MODULE(tensor_bindings, m) {
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

    m.def("tensor", [](std::vector<int> sizes, std::vector<float> values) {
        // cast the function pointer to resolve the ambiguity
        auto func = static_cast<std::shared_ptr<cg::Tensor> (*)(std::vector<int>, std::vector<float>)>(&cg::tensor);
        return func(sizes, values);
    }, R"pbdoc(Magic tensor)pbdoc",
    py::arg("sizes"), py::arg("values"));

    // TODO(yrom1): When doing C++ bindings does repr
    //              override str when no str present?
    py::class_<cg::Tensor, std::shared_ptr<cg::Tensor>>(m, "Tensor")
        .def(py::init<std::vector<int>, std::vector<float>>())
        .def("get_shared", &cg::Tensor::get_shared)
        .def("backward", &cg::Tensor::backward)
        .def("zero_grad", &cg::Tensor::zero_grad)
        .def("sum", &cg::Tensor::sum)
        .def("relu", &cg::Tensor::relu)
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
        .def("__matmul__", [](std::shared_ptr<cg::Tensor> a, std::shared_ptr<cg::Tensor> b) { return a.get()->matmul(b); })
        .def("__str__", [](std::shared_ptr<cg::Tensor> t) {
            std::ostringstream os;
            os << t;
            return os.str();
        })
        .def("__repr__", [](std::shared_ptr<cg::Tensor> t) {
            return t.get()->repr();
        });

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
