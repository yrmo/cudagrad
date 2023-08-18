import os
import re
import shutil
import sqlite3
import subprocess
from functools import wraps
from getpass import getpass
from subprocess import run
from timeit import timeit

import fire
import toml

# imports for timeit below
# ignore'ing because type checkers dont know this
import torch  # type: ignore

import cudagrad as cg  # type: ignore

CPP_FILES = "./tests/test.cpp ./src/tensor.hpp ./src/ops.cu"

cmd1 = ("import cudagrad as cg;", "a = cg.tensor([2, 2], [2.0, 3.0, 4.0, 5.0]); b = cg.tensor([2, 2], [6.0, 7.0, 8.0, 9.0]); c = a @ b", "tiny matmul")
cmd2 = ("import torch;", "a = torch.tensor(((2.0, 3.0), (4.0, 5.0))); b = torch.tensor(((6.0, 7.0), (8.0, 9.0))); c = a @ b", "tiny matmul")
cmd3 = ("import cudagrad as cg;", "a = cg.tensor([2, 2], [2.0, 3.0, 4.0, 5.0]); b = cg.tensor([2, 2], [6.0, 7.0, 8.0, 9.0]); c = a @ b; d = c.sum(); d.backward()", "tiny backward")
cmd4 = ("import torch;", "a = torch.tensor(((2.0, 3.0), (4.0, 5.0)), requires_grad=True); b = torch.tensor(((6.0, 7.0), (8.0, 9.0)), requires_grad=True); c = a @ b; d = c.sum(); d.backward()", "tiny backward")
cmd5 = ("import numpy as np", "a = np.array([[2.0, 3.0],[4.0, 5.0]]); b = np.array([[6.0, 7.0], [8.0, 9.0]]); c = a @ b;", "tiny matmul")
TESTS = [cmd1, cmd2, cmd3, cmd4, cmd5]

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


class Makefile:
    def __init__(self):
        global CPP_FILES
        for file in CPP_FILES.split():
            assert os.path.isfile(f"{file}") == True

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
        """
        create database performance;

        create table tests(
            id serial PRIMARY KEY,
            key varchar(255),
            setup varchar(255),
            statement varchar(1000),
            version varchar(255),
            loop_nanoseconds real,
            created_at timestamp DEFAULT current_timestamp
        );

        insert into tests(key, setup, statement, version, loop_nanoseconds) values(
            'tiny matmul'
            ,'import cudagrad as cg'
            ,'a = cg.tensor([2, 2], [2.0, 3.0, 4.0, 5.0]); b = cg.tensor([2, 2], [6.0, 7.0, 8.0, 9.0]); c = a @ b'
            ,'2.0.0'
            ,'1.77'
        );
        """
        def get_password(self) -> str:
            return getpass()

        def connect(self):
            run(f"sqlite3 {DATABASE}")

        def insert_test_record(self, *args):
            connection = sqlite3.connect(DATABASE)
            cursor = connection.cursor()
            query = """
            INSERT INTO tests (key, setup, statement, version, loop_nanoseconds)
            VALUES (?, ?, ?, ?, ?);
            """
            cursor.execute(query, args)
            connection.commit()
            cursor.close()
            connection.close()

        def run_tests(self) -> None:
            for test in TESTS:
                print(test)
                statement = test[1]
                setup = test[0]
                key = test[2]
                version = None # TODO
                loop_nanoseconds = timeit(statement, setup)
                self.insert_test_record(key, setup, statement, version, loop_nanoseconds)
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
    fire.Fire(Makefile)
