-- script that displays the max temperature of each state (ordered by State name)

select state, max(value) from temperatures group by state;