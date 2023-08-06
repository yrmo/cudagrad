import os
import re
import shutil
import subprocess
from functools import wraps

import fire
import toml
import torch

CPP_FILES = "./tests/test.cpp ./src/tensor.hpp ./src/ops.cu"


def echo(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        print(args[0])
        function(*args, **kwargs)

    return wrapper


@echo
def RUN(input: str) -> None:
    os.system(input)


class Makefile:
    def lint(self):

        RUN(f"python -m cpplint {CPP_FILES}")
        RUN("python -m mypy --exclude build --ignore-missing-imports --pretty .")

    def clean(self):

        RUN("python -m isort .")
        RUN("python -m black .")
        RUN("clang-format -i -style=Google {CPP_FILES}")

    def test(self):
        @echo
        def RUN(input: str) -> None:
            subprocess.check_call(input, shell=True)

        RUN("git submodule update --init --recursive")
        if os.path.exists("build"):
            shutil.rmtree("build")
        os.makedirs("build", exist_ok=True)
        os.chdir("build")
        RUN("cmake -DCMAKE_PREFIX_PATH=" + torch.utils.cmake_prefix_path + " ..")
        RUN("cmake ..")
        RUN("make")
        RUN("./tensor_test")
        RUN("python -m pip uninstall -y cudagrad")
        RUN("python -m pip cache purge")
        os.chdir(os.path.expanduser("~/cudagrad"))
        RUN("python -m pip install .")
        RUN("python tests/test.py")
        os.chdir(os.path.expanduser("~/cudagrad"))
        RUN("c++ -std=c++11 -I./src examples/example.cpp && ./a.out")
        RUN("python -m pip install cudagrad")
        RUN("python ./examples/example.py")

    def publish(self):
        RUN = os.system
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


if __name__ == "__main__":
    fire.Fire(Makefile)
