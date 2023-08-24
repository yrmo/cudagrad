PRAGMA foreign_keys=OFF;
BEGIN TRANSACTION;
CREATE TABLE test (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key VARCHAR(255),
            setup VARCHAR(255),
            statement VARCHAR(1000)
        );
INSERT INTO test VALUES(1,'tiny matmul','import cudagrad as cg;','a = cg.tensor([2, 2], [2.0, 3.0, 4.0, 5.0]); b = cg.tensor([2, 2], [6.0, 7.0, 8.0, 9.0]); c = a @ b');
INSERT INTO test VALUES(2,'tiny matmul','import torch;','a = torch.tensor(((2.0, 3.0), (4.0, 5.0))); b = torch.tensor(((6.0, 7.0), (8.0, 9.0))); c = a @ b');
INSERT INTO test VALUES(3,'tiny backward','import cudagrad as cg;','a = cg.tensor([2, 2], [2.0, 3.0, 4.0, 5.0]); b = cg.tensor([2, 2], [6.0, 7.0, 8.0, 9.0]); c = a @ b; d = c.sum(); d.backward()');
INSERT INTO test VALUES(4,'tiny backward','import torch;','a = torch.tensor(((2.0, 3.0), (4.0, 5.0)), requires_grad=True); b = torch.tensor(((6.0, 7.0), (8.0, 9.0)), requires_grad=True); c = a @ b; d = c.sum(); d.backward()');
INSERT INTO test VALUES(5,'tiny matmul','import numpy as np','a = np.array([[2.0, 3.0],[4.0, 5.0]]); b = np.array([[6.0, 7.0], [8.0, 9.0]]); c = a @ b;');
CREATE TABLE result (
            id INTEGER,
            version VARCHAR(255),
            loop_nanoseconds REAL,
            created_at timestamp DEFAULT current_timestamp,
            FOREIGN KEY (id) REFERENCES test (id)
        );
INSERT INTO result VALUES(1,NULL,1.8553488750185351818,'2023-08-19 15:48:28');
INSERT INTO result VALUES(2,NULL,5.0702016249997541308,'2023-08-19 15:48:33');
INSERT INTO result VALUES(3,NULL,2.6094079999893438071,'2023-08-19 15:48:36');
INSERT INTO result VALUES(4,NULL,22.380717291001928082,'2023-08-19 15:48:58');
INSERT INTO result VALUES(5,NULL,1.7245115420082584023,'2023-08-19 15:49:00');
INSERT INTO result VALUES(1,NULL,1.7864600839966442435,'2023-08-19 15:50:30');
INSERT INTO result VALUES(2,NULL,5.0520202500047162175,'2023-08-19 15:50:35');
INSERT INTO result VALUES(3,NULL,2.6031652079836931079,'2023-08-19 15:50:38');
INSERT INTO result VALUES(4,NULL,22.451889082993146118,'2023-08-19 15:51:01');
INSERT INTO result VALUES(5,NULL,1.7419124160078354179,'2023-08-19 15:51:02');
CREATE TABLE pip_compile_speed (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            seconds REAL,
            created_at timestamp DEFAULT current_timestamp
        );
INSERT INTO pip_compile_speed VALUES(1,6.8138802051544189453,'2023-08-21 16:27:51');
INSERT INTO pip_compile_speed VALUES(2,6.5523221492767333984,'2023-08-21 16:28:27');
DELETE FROM sqlite_sequence;
INSERT INTO sqlite_sequence VALUES('test',5);
INSERT INTO sqlite_sequence VALUES('pip_compile_speed',2);
COMMIT;
