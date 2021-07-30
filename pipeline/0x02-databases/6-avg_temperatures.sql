--script that displays the average temperature (Fahrenheit) by city ordered by temperature (descending).

select city, AVG(value) as avg_temp from temperatures group by city ORDER by avg_temp DESC;