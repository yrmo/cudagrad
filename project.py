import os
import re
import shutil
import sqlite3
import subprocess
import time
from functools import wraps
from getpass import getpass
from itertools import chain
from pathlib import Path
from subprocess import run
from timeit import timeit

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
            assert os.path.isfile(f"{file}") == True

    def connect(self):
        RUN("ssh ryan@192.168.0.28")

    def code(self):
        RUN(
            'code --folder-uri "vscode-remote://ssh-remote+ryan@192.168.0.28/home/ryan/cudagrad"'
        )

    def github_release(self):
        RUN("python makefile.py bump patch")
        RUN(
            "version=$(python -c \"import toml; print(toml.load('pyproject.toml')['project']['version'])\")"
        )
        RUN('git tag -a $version -m "Release version $version"')
        RUN("git add cudagrad/__init__.py pyproject.toml")
        RUN("git commit -m $version")
        RUN("git push origin $version")
        RUN('echo "Version $version has been tagged and pushed to GitHub."')
        RUN('file_name="release_notes.txt"')
        RUN("code --wait $file_name")
        RUN("release_notes=$(cat $file_name)")
        RUN('echo "Content captured:"')
        RUN('echo "$release_notes"')
        RUN('gh release create $version -t $version -n "$release_notes"')
        RUN('echo "GitHub release for version $version has been created."')
        RUN("python makefile.py publish")
        RUN("rm $file_name")

    def lint(self):
        RUN(f"python -m cpplint {CPP_FILES}")
        RUN("python -m mypy --exclude build --ignore-missing-imports --pretty .")

    def clean(self):
        RUN("python -m isort .")
        RUN("python -m black .")
        RUN(f"clang-format -i -style=Google {CPP_FILES}")

    def SO(self):
        RUN("pip install .")
        RUN(
            "cp build/lib.macosx-13.2-arm64-cpython-311/cudagrad/tensor.cpython-311-darwin.so ./cudagrad/"
        )

    def test(self, processor):
        @echo
        def RUN(input: str) -> None:
            subprocess.check_call(input, shell=True)

        if processor == "CPU":
            RUN("cp CMakeListsCPU.txt CMakeLists.txt")
        elif processor == "CUDA":
            RUN("cp CMakeListsCUDA.txt CMakeLists.txt")
        else:
            raise ValueError(f"Unknown option for {processor=}!")

        RUN("git submodule update --init --recursive")
        if os.path.exists("build"):
            shutil.rmtree("build")
        os.makedirs("build", exist_ok=True)
        os.chdir("build")
        RUN("cmake -DCMAKE_PREFIX_PATH=" + torch.utils.cmake_prefix_path + " ..")
        RUN("cmake ..")
        RUN("make")
        RUN("./tensor_test")
        # FIXME skipping installation 'tests' on github runner for now
        if (str(Path(".").resolve()).split("/")[2]) == "runner":
            return
        RUN("python -m pip uninstall -y cudagrad")
        RUN("python -m pip cache purge")
        os.chdir(os.path.expanduser("~/cudagrad"))
        RUN("rm CMakeLists.txt")
        RUN("python -m pip install .")
        RUN(
            "cp ./build/lib.macosx-13.2-arm64-cpython-311/cudagrad/tensor.cpython-311-darwin.so ./cudagrad"
        )
        RUN("python tests/test_backward.py")
        os.chdir(os.path.expanduser("~/cudagrad"))
        RUN("python -m pip install cudagrad")
        RUN("python -m cudagrad.linear")
        RUN("python -m cudagrad.mlp")
        RUN("git restore cudagrad/plots/*.jpg")

    def test_python_3_7(self):
        RUN("pyenv global 3.7")
        RUN("pip uninstall -y cudagrad")
        RUN("cd /")
        RUN("pip install cudagrad")
        RUN('python -c "import cudagrad as cg; print(cg.tensor([1], [4.2]))"')
        RUN("cd ~/cudagrad")

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

    class DB:
        def connect(self):
            run(f"sqlite3 {DATABASE}")

        def pip_speed(self):
            start_time = time.time()
            process = subprocess.run(
                ["pip", "install", "."], stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            end_time = time.time()
            elapsed_time = end_time - start_time
            if process.returncode == 0:
                print("Installation succeeded!")
            else:
                print("Installation failed!")
                print("Error output:", process.stderr.decode())
            print("Elapsed time:", elapsed_time, "seconds")
            self.insert_into_pip(elapsed_time)

        # we're getting close to needing generic insert for DRY, but not yet

        def insert_into_pip(self, speed):
            connection = sqlite3.connect(DATABASE)
            cursor = connection.cursor()
            query = """
            INSERT INTO pip_compile_speed (id, seconds)
            VALUES (?, ?);
            """
            cursor.execute(query, (None, speed))
            connection.commit()
            cursor.close()
            connection.close()

        def insert_into_test(self, key, setup, statement):
            connection = sqlite3.connect(DATABASE)
            cursor = connection.cursor()
            query = """
            INSERT INTO test (id, key, setup, statement)
            VALUES (?, ?, ?, ?);
            """
            cursor.execute(query, (None, key, setup, statement))
            connection.commit()
            cursor.close()
            connection.close()

        def _insert_into_result(self, *args):
            connection = sqlite3.connect(DATABASE)
            cursor = connection.cursor()
            query = """
            INSERT INTO result (id, version, loop_nanoseconds)
            VALUES (?, ?, ?);
            """
            cursor.execute(query, args)
            connection.commit()
            cursor.close()
            connection.close()

        def run_tests(self) -> None:
            connection = sqlite3.connect(DATABASE)
            cursor = connection.cursor()
            query = """
            SELECT *
            FROM test
            """
            cursor.execute(query)
            tests = cursor.fetchall()
            cursor.close()
            connection.close()

            for test in tests:
                number = test[4]
                statement = test[3]
                setup = test[2]
                key = test[1]
                id = test[0]
                version = None  # TODO

                print(f"Running performance test:\n{statement}")
                print(f"{number=}!")
                loop_nanoseconds = timeit(statement, setup, number=number) / number

                self._insert_into_result(id, version, loop_nanoseconds)

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
