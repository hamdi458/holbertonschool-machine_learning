-- creates a stored procedure ComputeAverageScoreForUser that computes and store the average score for a student.

DELIMITER $$

CREATE PROCEDURE ComputeAverageScoreForUser (IN user_id INT)
BEGIN

SELECT AVG(score) INTO @sc FROM corrections WHERE corrections.user_id = user_id;

UPDATE users
SET average_score = @sc
WHERE id = user_id;
END $$
DELIMITER ;