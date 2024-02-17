# Available at setup time due to pyproject.toml

import toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import find_packages, setup


import os
import re
import subprocess
import sys
import platform
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)


# assert which("nvcc") is not None


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
    ext_modules=[CMakeExtension('cudagrad')],
    # extras_require={"test": "pytest"},
    # Currently, build_ext only provides an optional "highest supported C++
    # level" feature, but in the future it may provide more features.
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    python_requires=">=3.7",
    packages=find_packages(),
    include_package_data=True,
)
