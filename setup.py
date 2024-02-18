import os
import subprocess
import sys
from shutil import move

import toml
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

from pybind11.setup_helpers import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build tensor!")

        for extension in self.extensions:
            self.build_extension(extension)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args
                              , cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.'], cwd=self.build_temp)

        # print("Listing all files in build_temp directory:")
        # for root, dirs, files in os.walk(self.build_temp):
        #     for file in files:
        #         print(os.path.join(root, file))

        built_lib = os.path.join(self.build_temp,'lib', 'tensor.so')
        dest_lib = os.path.join(self.build_lib, 'cudagrad', 'tensor.so')
        move(built_lib, dest_lib)


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
    url="https://github.com/cudagrad/cudagrad",
    description="A tensor-valued autograd engine for Python",
    long_description="",
    ext_modules=[CMakeExtension('cudagrad.tensor')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    python_requires=">=3.7",
    packages=find_packages(),
    include_package_data=True,
)
