create database performance;

create table tests(
  id serial,
  setup varchar(255),
  statement varchar(1000),
  version varchar(255),
  loop_nanoseconds real,
  created_at timestamp DEFAULT current_timestamp
);

insert into tests(setup, statement, version, loop_nanoseconds) values(
  'import cudagrad as cg',
  'a = cg.tensor([2, 2], [2.0, 3.0, 4.0, 5.0]); b = cg.tensor([2, 2], [6.0, 7.0, 8.0, 9.0]); c = a @ b',
  '2.0.0',
  '1.77'
);
