# type: ignore

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


@echo
def CHECK(input: str) -> None:
    subprocess.check_call(input, shell=True)


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
        EXCLUDE = (
            "--exclude build --exclude cccl --exclude pybind11 --exclude googletest"
        )
        subprocess.run(f"python -m mypy {EXCLUDE} .", shell=True, capture_output=True)
        CHECK("python -m mypy --install-types --non-interactive")
        CHECK(f"python -m cpplint {CPP_FILES}")
        CHECK(f"python -m mypy {EXCLUDE} --ignore-missing-imports --pretty .")
        CHECK(f"ruff check {EXCLUDE} .")

    def format(self):
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

        process = subprocess.run(
            ["nvcc", "-x", "cu", "-o", TEST_CUDA_FILENAME, "-"],
            input=code,
            text=True,
            capture_output=True,
        )

        if process.returncode != 0:
            print(process.stderr)
        else:
            run_command = [f"./{TEST_CUDA_FILENAME}"]
            run_process = subprocess.run(run_command, capture_output=True, text=True)

            if run_process.returncode == 0:
                print(run_process.stdout)
            else:
                print(repr(run_process.stderr))

            assert run_process.stdout == "no error"
            os.remove(TEST_CUDA_FILENAME)

    def test(self):
        CWD = os.getcwd()

        self._test_cuda_setup()
        CHECK("nvcc tests/test_setup.cu && ./a.out")
        CHECK("rm a.out")

        CHECK("git submodule update --init --recursive")
        if os.path.exists("build"):
            shutil.rmtree("build")
        os.makedirs("build", exist_ok=True)
        os.chdir("build")
        CHECK(
            "cmake -DCMAKE_PREFIX_PATH="
            + torch.utils.cmake_prefix_path
            + f" {Path('../tests').resolve()}"
        )
        CHECK("cmake ../tests")
        CHECK("make")
        CHECK("./tensor_test")
        # FIXME skipping installation 'tests' on github runner for now
        if (str(Path(".").resolve()).split("/")[2]) == "runner":
            return

        os.chdir("..")

        CHECK("python -m pip uninstall -y cudagrad")
        CHECK("python -m pip cache purge")
        os.chdir(CWD)
        CHECK("pip install .")
        CHECK("python tests/test_backward.py")
        CHECK("python tests/test_forward.py")
        CHECK("python ./examples/or.py")
        CHECK("python ./examples/xor.py")
        CHECK("python ./examples/moons.py")
        CHECK("python ./examples/mnist.py")
        CHECK("git restore examples/plots/*.jpg")

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

    def _profiles_total_tt(self) -> list[tuple([str, float])]:
        from pathlib import Path
        from pstats import Stats

        p = Path("./examples/profiles")
        ans = []
        for profile in p.glob("*"):
            ans.append(tuple([profile.stem, Stats(str(profile.resolve())).total_tt]))
        return sorted(ans, key=lambda x: x[-1])

    def profiles_markdown_table(self) -> str:
        data = self._profiles_total_tt()
        header = "| Dataset | Time (seconds) |\n"
        divider = "|---------|----------------|\n"
        table = header + divider
        for dataset, time in data:
            row = f"| {dataset} | {time:.2f} |\n"
            table += row
        return table

    def release(self):
        self.bump("patch")
        # TODO
        """\
        version=$(python -c "import toml; print(toml.load('pyproject.toml')['project']['version'])")
        git tag -a $version -m "Release version $version"
        git add cudagrad/__init__.py pyproject.toml
        git commit -m $version
        git push origin $version
        echo "Version $version has been tagged and pushed to GitHub."
        file_name="release_notes.txt"
        code --wait $file_name
        release_notes=$(cat $file_name)
        echo "Content captured:"
        echo "$release_notes"
        gh release create $version -t $version -n "$release_notes"
        echo "GitHub release for version $version has been created."
        """
        self.publish()
        """
        rm $file_name
        """

    def prepublish(self):
        # TODO
        """
        rm -rf dist
        python setup.py sdist bdist_wheel
        cd dist
        pip install cudagrad-0.1.0.tar.gz # TODO fix harcode
        python -c "from cudagrad import hello; hello()"
        """

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
