
To compile:
	make

To run:
	./raycaster volume.raw gridx gridy gridz colormap.raw length_colormap dimx dimy dimz

To run test image of sphere map:
	./raycaster sphere gridx gridy gridz colormap.raw length_colormap
	grid values should be equal. using small colormap gives easy to see serrated color.

Included COLORMAPS
autumn.raw 1000
cool.raw 1000
bone.raw 1000
jet.raw 1000
cool_small.raw 10 //good for sphere map
Generated using python script nCr.py.

KEYBOARD COMMANDS

q,e - Move light -,+ in z direction
w,s - Move light +,- in y direction
a,d - Move light -,+ in x direction

z,Z - zoom out, zoom in

S - screen shot. Save as screenshot.ppm
A - screen shot first image to take difference of. Save A as diff_A.ppm
B - screen shot second image to take difference with A. Save B as diff_B.ppm.
	Calculate thresholded difference image. Save as diff_im.ppm
	Prints out Average L2 difference. 

0,1,2 - view in orthogonal direction in x,y,z planes

l/L - switch light off and on

n,t,c - switch between nearest, trilinear and tricubic interpolation. Default is trilinear.

T - Change opacity min max opacity thresholds. All values above range are considered opaque. All values below range are 	considered completely transparent.
Y - Change sampling range from colormap. Clamp all opacity values to within this range. 

i/I - decrement and increment the front-to-back steps in ray caster. Step value below 0 are considered as no threshold. Instead the recursive term's coefficient is thresholded. 







