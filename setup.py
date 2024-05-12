import os
import subprocess
import sys
from shutil import move

import toml
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

from pybind11.setup_helpers import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DPYTHON_EXECUTABLE=" + sys.executable,
        ]

        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", ".", "--target", ext.name] + build_args,
            cwd=self.build_temp,
        )
        move(
            os.path.join(self.build_lib, "tensor.so"),
            os.path.join(extdir, "cudagrad", "tensor.so"),
        )


def get_version_from_toml():
    data = toml.load("pyproject.toml")
    version = data.get("project", {}).get("version", None)
    if version is None:
        raise RuntimeError("Can't get version in TOML!")
    else:
        return version


__version__ = get_version_from_toml()

setup(
    name="cudagrad",
    version=__version__,
    author="Ryan Moore",
    author_email="ryanm.inbox@gmail.com",
    url="https://github.com/yrmo/cudagrad",
    description="A tensor-valued autograd engine for Python",
    long_description=open("README.md").read(),
    ext_modules=[CMakeExtension("tensor")],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    python_requires=">=3.7",
    packages=find_packages(),
    include_package_data=True,
)
