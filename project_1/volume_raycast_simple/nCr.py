import math
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

import matplotlib
print matplotlib.__version__

def get_colormap():
	x = np.linspace(0.0, 1.0, 10)
	map_name = 'cool'
	rgb = cm.get_cmap(map_name)(x)
	rgb = rgb/np.max(rgb,0,keepdims=True);
	rgb[rgb!=rgb]=0;
	# rgb[:,3]=rgb[:,3]*255
	out_file = map_name+'_small.raw';
	print rgb[0]
	print np.max(rgb,0)
	vals = rgb.flatten()*255
	# /256.
	vals = [np.uint8(val) for val in vals]
	# print vals[:4]
	# print rgb[0]

	# print vals[10*4:10*4+4]
	# print rgb[10]
	
	val_b = bytearray(vals);

	f = open(out_file,'wb')
	f.write(val_b)
	f.close()

	# print vals[:100]

	# print rgb.shape
	# print rgb[500]
	# print np.min(rgb,0)
	# print np.max(rgb,0)
	# cmap = plt.get_cmap('autumn')

	# print cmap

def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

def main():
	get_colormap()
	# n = 2
	# for r in range(n+1):
	# 	print n,'choose',r,'=',nCr(n,r);

if __name__=='__main__':
	main();
