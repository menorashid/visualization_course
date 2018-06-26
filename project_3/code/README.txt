README

sample command:

python voronoi.py --scatter_fun serrated_sphere --resolution 16

for more detail on options run

python voronoi.py --h 


For every iteration, the mean squared distance with the ground truth is calculated for both nearest and sisbson methods and printed out

you can change the number of pts for the voronoi ('double' or 'add'), and the resolution while the program is running ('resolution').

There are 2 types of data functions 'sphere', and 'sphere_serrated'

