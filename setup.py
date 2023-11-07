# Available at setup time due to pyproject.toml

from os import environ
from shutil import which

import toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import find_packages, setup

# TEMP: long term we remove this, we will not support people without nvcc
# if which("nvcc") is not None:
#     environ["CXX"] = "nvcc"


def get_version_from_toml():
    data = toml.load("pyproject.toml")
    version = data.get("project", {}).get("version", None)
    if version is None:
        raise RuntimeError("Can't get version in TOML!")
    else:
        return version


__version__ = get_version_from_toml()

# The main interface is through Pybind11Extension.
# * You can add cxx_std=11/14/17, and then build_ext can be removed.
# * You can set include_pybind11=false to add the include directory yourself,
#   say from a submodule.
#
# Note:
#   Sort input source files if you glob sources to ensure bit-for-bit
#   reproducible builds (https://github.com/pybind/cudagrad/pull/53)

ext_modules = [
    Pybind11Extension(
        "cudagrad.tensor",
        ["src/bindings.cpp"],
        # Example: passing in the version to the compiled code
        define_macros=[("VERSION_INFO", __version__)],
    ),
]

setup(
    name="cudagrad",
    version=__version__,
    author="Ryan Moore",
    author_email="ryanm.inbox@gmail.com",
    url="https://github.com/cudagrad/cudagrad",
    description="A small tensor-valued autograd engine",
    long_description="",
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    # Currently, build_ext only provides an optional "highest supported C++
    # level" feature, but in the future it may provide more features.
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
    packages=find_packages(),
    include_package_data=True,
)
