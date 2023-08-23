CREATE TABLE sqlite_sequence(name,seq);
CREATE TABLE test (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key VARCHAR(255),
            setup VARCHAR(255),
            statement VARCHAR(1000)
        );
CREATE TABLE result (
            id INTEGER,
            version VARCHAR(255),
            loop_nanoseconds REAL,
            created_at timestamp DEFAULT current_timestamp,
            FOREIGN KEY (id) REFERENCES test (id)
        );
CREATE TABLE pip_compile_speed (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            seconds REAL,
            created_at timestamp DEFAULT current_timestamp
        );
