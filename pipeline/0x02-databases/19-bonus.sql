-- procedure AddBonus that adds a new correction for a student.

DELIMITER $$

CREATE OR REPLACE PROCEDURE AddBonus (IN user_id INT,
IN project_name VARCHAR(255),
IN score INT)
BEGIN
IF NOT EXISTS (SELECT id FROM projects WHERE name = project_name) THEN
    INSERT INTO projects(name) VALUES (project_name);
END IF;
SELECT id INTO @id_of_project FROM projects WHERE name = project_name;
-- SET @id_of_project = (SELECT id FROM projects WHERE name = project_name);   

INSERT INTO corrections(user_id, project_id, score)
            VALUES(user_id, @id_of_project, score);
END $$
DELIMITER ;