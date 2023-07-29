import os
import shutil

import fire
import toml
import torch
from pip._internal.cli import main as pip_main

run = os.system


class Makefile:
    def lint(self):
        run("python -m mypy --ignore-missing-imports --pretty .")

    def clean(self):
        run("python -m isort .")
        run("python -m black .")

    def test(self):
        run("git submodule update --init --recursive")
        if os.path.exists("build"):
            shutil.rmtree("build")
        os.makedirs("build", exist_ok=True)
        os.chdir("build")
        run("cmake -DCMAKE_PREFIX_PATH=" + torch.utils.cmake_prefix_path + " ..")
        run("cmake ..")
        run("make")
        run("./cudagrad_test")
        pip_main(["uninstall", "-y", "cudagrad"])
        pip_main(["cache", "purge"])
        os.chdir(os.path.expanduser("~/cudagrad"))
        pip_main(["install", "."])
        run("python tests/test.py")
        os.chdir(os.path.expanduser("~/cudagrad"))
        run("c++ -std=c++11 -I./src examples/example.cpp && ./a.out")
        pip_main(["install", "cudagrad"])
        run("python ./examples/example.py")

    def publish(self):
        pip_main(["uninstall", "-y", "cudagrad"])
        pip_main(["cache", "purge"])
        if os.path.exists("dist"):
            shutil.rmtree("dist")
        run("python setup.py sdist")
        pip_main(["install", "--upgrade", "twine"])
        run("python -m twine upload dist/*")

    def bump(self, version_type):
        d = toml.load("pyproject.toml")
        version_list = d["project"]["version"].split(".")
        if version_type == "major":
            version_list[0] = str(int(version_list[0]) + 1)
        elif version_type == "minor":
            version_list[1] = str(int(version_list[1]) + 1)
        elif version_type == "patch":
            version_list[2] = str(int(version_list[2]) + 1)
        else:
            raise ValueError(f"Invalid bump option: `{version_type}`.")
        d["project"]["version"] = ".".join(version_list)
        toml.dump(d, open("pyproject.toml", "w"))


if __name__ == "__main__":
    fire.Fire(Makefile)
