from subprocess import run

import boto3


def main():
  rds_client = boto3.client('rds')
  db_instance = rds_client.describe_db_instances(DBInstanceIdentifier='boto-postgres')['DBInstances'][0]

  HOSTNAME = db_instance['Endpoint']['Address']
  PORT = db_instance['Endpoint']['Port']
  USERNAME = "ryan"
  DATABASE = "postgres"

  run(f"psql -h {HOSTNAME} -p {PORT} -U {USERNAME} -d {DATABASE} -W", shell=True)

if __name__ == '__main__':
  main()
