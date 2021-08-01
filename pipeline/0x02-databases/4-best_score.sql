-- lists all records with a score >= 10 in the table second_table in your MySQL server
-- Records should be ordered by score (top first)

select score, name from second_table where score >= 10 ORDER BY score DESC;