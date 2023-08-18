import sqlite3
from getpass import getpass
from subprocess import run
from timeit import timeit

import fire

# imports for timeit, possibly dynamically (?)
import torch  # type: ignore

import cudagrad as cg  # type: ignore

cmd1 = ("import cudagrad as cg;", "a = cg.tensor([2, 2], [2.0, 3.0, 4.0, 5.0]); b = cg.tensor([2, 2], [6.0, 7.0, 8.0, 9.0]); c = a @ b", "tiny matmul")
cmd2 = ("import torch;", "a = torch.tensor(((2.0, 3.0), (4.0, 5.0))); b = torch.tensor(((6.0, 7.0), (8.0, 9.0))); c = a @ b", "tiny matmul")
cmd3 = ("import cudagrad as cg;", "a = cg.tensor([2, 2], [2.0, 3.0, 4.0, 5.0]); b = cg.tensor([2, 2], [6.0, 7.0, 8.0, 9.0]); c = a @ b; d = c.sum(); d.backward()", "tiny backward")
cmd4 = ("import torch;", "a = torch.tensor(((2.0, 3.0), (4.0, 5.0)), requires_grad=True); b = torch.tensor(((6.0, 7.0), (8.0, 9.0)), requires_grad=True); c = a @ b; d = c.sum(); d.backward()", "tiny backward")
cmd5 = ("import numpy as np", "a = np.array([[2.0, 3.0],[4.0, 5.0]]); b = np.array([[6.0, 7.0], [8.0, 9.0]]); c = a @ b;", "tiny matmul")
TESTS = [cmd1, cmd2, cmd3, cmd4, cmd5]

class DB:
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


if __name__ == '__main__':
  DATABASE = "performance.db"
  fire.Fire(DB)
