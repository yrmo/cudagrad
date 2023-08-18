from getpass import getpass
from subprocess import run
from timeit import timeit

import boto3
import fire
import psycopg2

# imports for timeit, possibly dynamically (?)
import torch  # type: ignore

import cudagrad as cg  # type: ignore

cmd1 = ("import cudagrad as cg;", "a = cg.tensor([2, 2], [2.0, 3.0, 4.0, 5.0]); b = cg.tensor([2, 2], [6.0, 7.0, 8.0, 9.0]); c = a @ b", "tiny matmul")
cmd2 = ("import torch;", "a = torch.tensor(((2.0, 3.0), (4.0, 5.0))); b = torch.tensor(((6.0, 7.0), (8.0, 9.0))); c = a @ b", "tiny matmul")
cmd3 = ("import cudagrad as cg;", "a = cg.tensor([2, 2], [2.0, 3.0, 4.0, 5.0]); b = cg.tensor([2, 2], [6.0, 7.0, 8.0, 9.0]); c = a @ b; d = c.sum(); d.backward()", "tiny matmul")
cmd4 = ("import torch;", "a = torch.tensor(((2.0, 3.0), (4.0, 5.0)), requires_grad=True); b = torch.tensor(((6.0, 7.0), (8.0, 9.0)), requires_grad=True); c = a @ b; d = c.sum(); d.backward()", "tiny matmul")
cmd5 = ("import numpy as np", "a = np.array([[2.0, 3.0],[4.0, 5.0]]); b = np.array([[6.0, 7.0], [8.0, 9.0]]); c = a @ b;", "tiny matmul")
TESTS = [cmd1, cmd2, cmd3, cmd4, cmd5]

global PASSWORD

class DB:
  def get_password(self) -> str:
    return getpass()

  def connect(self):
    run(f"psql -h {HOSTNAME} -p {PORT} -U {USERNAME} -d {DATABASE} -W", shell=True)

  def insert_test_record(self, *args):
      global PASSWORD
      connection = psycopg2.connect(
          host=HOSTNAME,
          port=PORT,
          user=USERNAME,
          password=PASSWORD,
          dbname=DATABASE
      )
      cursor = connection.cursor()
      query = """
      INSERT INTO tests (key, setup, statement, version, loop_nanoseconds)
      VALUES (%s, %s, %s, %s, %s);
      """
      cursor.execute(query, args)
      connection.commit()
      cursor.close()
      connection.close()

  def run_tests(self) -> None:
    global PASSWORD
    PASSWORD = self.get_password()
    for test in TESTS:
      print(test)
      statement = test[1]
      setup = test[0]
      key = test[2]
      version = "NULL" # TODO
      loop_nanoseconds = timeit(statement, setup)
      self.insert_test_record(key, setup, statement, version, loop_nanoseconds)


if __name__ == '__main__':
  rds_client = boto3.client('rds')
  db_instance = rds_client.describe_db_instances(DBInstanceIdentifier='boto-postgres')['DBInstances'][0]

  HOSTNAME = db_instance['Endpoint']['Address']
  PORT = db_instance['Endpoint']['Port']
  USERNAME = "ryan"
  DATABASE = "performance"

  fire.Fire(DB)
