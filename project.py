import os
import re
import shutil
import subprocess
from functools import wraps
from itertools import chain
from os import environ
from pathlib import Path

import fire
import torch

glob_cpp = "*[.cpp|.hpp|.cu]"
CPP_FILES = " ".join(
    [
        str(x)
        for x in chain(Path("./src").glob(glob_cpp), Path("./tests").glob(glob_cpp))
    ]
)

DATABASE = "performance.db"


def echo(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        print(args[0])
        function(*args, **kwargs)

    return wrapper


@echo
def RUN(input: str) -> None:
    os.system(input)


class Project:
    def __init__(self):
        global CPP_FILES
        for file in CPP_FILES.split():
            assert os.path.isfile(f"{file}") is True

    def install(self):
        RUN("git submodule update --init --recursive")
        RUN("rm -rf build")
        RUN("mkdir build")
        os.chdir("build")
        RUN("cmake ..")
        RUN("make")
        RUN("cp tensor.so ../cudagrad/")

    def lint(self):
        RUN(f"python -m cpplint {CPP_FILES}")
        RUN("python -m mypy --exclude build --ignore-missing-imports --pretty .")
        RUN("ruff .")

    def clean(self):
        RUN("python -m isort .")
        RUN("python -m black .")
        RUN(f"clang-format -i -style=Google {CPP_FILES}")

    def build(self):
        RUN("pip install .")
        RUN(
            "cp build/lib.macosx-13.2-arm64-cpython-311/cudagrad/tensor.cpython-311-darwin.so ./cudagrad/"
        )

    def _test_cuda_setup(self):
        code = """\
        #include <stdio.h>

        __global__ void foo() {}

        int main() {
        foo<<<1,1>>>();
        printf("%s", cudaGetErrorString(cudaGetLastError()));
        return 0;
        }
        """

        TEST_CUDA_FILENAME = "test_cuda_setup"

        process = subprocess.run(['nvcc', '-x', 'cu', '-o', TEST_CUDA_FILENAME, '-'], input=code, text=True, capture_output=True)

        if process.returncode != 0:
            print(process.stderr)
        else:
            run_command = [f'./{TEST_CUDA_FILENAME}']
            run_process = subprocess.run(run_command, capture_output=True, text=True)
            
            if run_process.returncode == 0:
                print(run_process.stdout)
            else:
                print(repr(run_process.stderr))
            
            assert run_process.stdout == "no error"
            os.remove(TEST_CUDA_FILENAME)

    def test(self, processor):
        @echo
        def RUN(input: str) -> None:
            subprocess.check_call(input, shell=True)

        if processor == "CUDA":
            self._test_cuda_setup()
            RUN("nvcc tests/test_setup.cu && ./a.out")
            RUN("rm a.out")

        if processor == "CPU":
            RUN("cp tests/CMakeListsCPU.txt tests/CMakeLists.txt")
        elif processor == "CUDA":
            RUN("cp tests/CMakeListsCUDA.txt tests/CMakeLists.txt")
        else:
            raise ValueError(f"Unknown option for {processor=}!")

        RUN("git submodule update --init --recursive")
        if os.path.exists("build"):
            shutil.rmtree("build")
        os.makedirs("build", exist_ok=True)
        os.chdir("build")
        RUN("cmake -DCMAKE_PREFIX_PATH=" + torch.utils.cmake_prefix_path + f" {Path('../tests').resolve()}")
        RUN("cmake ../tests")
        RUN("make")
        RUN("./tensor_test")
        # FIXME skipping installation 'tests' on github runner for now
        if (str(Path(".").resolve()).split("/")[2]) == "runner":
            return

        os.chdir("..")
        RUN("rm ./tests/CMakeLists.txt")

        RUN("python -m pip uninstall -y cudagrad")
        RUN("python -m pip cache purge")
        os.chdir(os.path.expanduser("~/cudagrad"))
        RUN("pip install .")
        # RUN("python tests/test_backward.py")
        RUN("python ./examples/or.py")
        RUN("python ./examples/xor.py")
        RUN("python ./examples/moons.py")
        RUN("python ./examples/mnist.py")
        RUN("git restore examples/plots/*.jpg")

    def test_python_3_7(self):
        RUN("pyenv global 3.7")
        RUN("pip uninstall -y cudagrad")
        RUN("cd /")
        RUN("pip install cudagrad")
        RUN('python -c "import cudagrad as cg; print(cg.tensor([1], [4.2]))"')
        RUN("cd ~/cudagrad")

    def publish(self):
        RUN = os.system
        nb = "Tensor.ipynb"
        assert os.isfile(nb)
        RUN(f"jupyter nbconvert --to notebook --execute --inplace {nb}")
        RUN("python -m pip uninstall -y cudagrad")
        RUN("python -m pip cache purge")
        if os.path.exists("dist"):
            shutil.rmtree("dist")
        RUN("python setup.py sdist")
        RUN("python -m pip install --upgrade twine")
        RUN("python -m twine upload dist/*")

    def bump(self, version_type):
        # __version__ is in two places for now
        # not good, but keeps things simpleish

        # TODO check if they get out of sync because I manually changed one like an idiot

        with open("pyproject.toml", "r+") as f:
            content = f.read()
            version_match = re.search(r'version = "(\d+)\.(\d+)\.(\d+)"', content)
            if version_match:
                major, minor, patch = map(int, version_match.groups())
                if version_type == "major":
                    major += 1
                elif version_type == "minor":
                    minor += 1
                elif version_type == "patch":
                    patch += 1
                else:
                    raise ValueError(f"Invalid bump option: `{version_type}`.")
                new_version = f'version = "{major}.{minor}.{patch}"'
                content = re.sub(r'version = "\d+\.\d+\.\d+"', new_version, content)
                f.seek(0)
                f.write(content)
                f.truncate()
            else:
                print("No version found in the file.")

        version_numbers = [major, minor, patch]
        print(version_numbers)
        with open("./cudagrad/__init__.py", "r") as f:
            init_contents = f.read()

        with open("./cudagrad/__init__.py", "w") as f:
            dot = "."
            f.write(
                re.sub(
                    r'__version__ = "\d+\.\d+\.\d+"',
                    f'__version__ = "{dot.join([str(x) for x in version_numbers])}"',
                    init_contents,
                )
            )

    def profile(self, model):
        environ["PROFILING"] = "1"
        RUN(f"python -m cProfile -o ./profiles/{model}.prof -m cudagrad.{model}")

    class CheckRequirementsBroken:
        def check_program(self, program) -> None:
            """Checks if the specified program is installed"""
            path = shutil.which(program)
            if path is None:
                print(f"{program} is not installed.")
            else:
                print(f"{program} is installed at {path}.")

        def run(self):
            # sudo apt install clang-format
            # python -m pip install cpplint
            # TODO cmake...
            programs = ["clang-format", "cpplint", "cmake"]
            for program in programs:
                self.check_program(program)


if __name__ == "__main__":
    fire.Fire(Project)
