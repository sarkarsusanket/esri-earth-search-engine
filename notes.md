## Changes we need to make as the queries are not being handled:

Problem:
- The parser couldnt handle query like "places which used to be a forest but now is buildings" just seraches for "forest to buildings". 
Solution:
- Need to make the parser able to search thoprugh multiple years, allow year as a argument in viison-high and low

Problem 2:
- Also look into the latency issues.
Soltuion:NA

Problem 3:
- when asked for highways it just seraches motorwsy, should search all the primary+sec+terti roads instaed.

Problem 4: MAJOR
- need to be able to return polygons, only returns points now.

Problem 5:
- need to understand that sometimes osm returns a point or line, and you cannot poass that as a region. you need to buffer that.

Problem 6:
- Need to put a cap on the max outputs to 1k

Problem 7:
-Reverse change detect not working: "find me buildings that have disappeared since 2020 in bakersfield"




