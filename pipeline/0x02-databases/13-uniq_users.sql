-- script that creates a table users following these requirements
CREATE TABLE IF NOT EXISTS users
    (id INT PRIMARY KEY NOT NULL AUTO_INCREMENT,
    email VARCHAR(255) NOT NULL UNIQUE,
    name VARCHAR(255));

