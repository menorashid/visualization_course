//sample commands

python interpolator.py --scatter_fun sphere --resolution 16 --colormap jet --interpolant_type local_hardy --num_k 3 --n_pts 100

python interpolator.py --scatter_fun ../../Volume\ Files/neghip_8_64.raw 64 --resolution 32 --colormap gray --interpolant_type hardy  --n_pts 1000

*****
usage: interpolator.py [-h] [--scatter_fun scatter_fun [scatter_fun ...]]
                       [--n_pts n_pts] [--interpolant_type interpolant_type]
                       [--resolution resolution] [--r_sq r_sq] [--num_k num_k]
                       [--colormap colormap] [--idx_slice idx_slice]

Interpolate scattered pts

optional arguments:
  -h, --help            show this help message and exit
  --scatter_fun scatter_fun [scatter_fun ...]
                        scatter function or file. enter resolution after
                        filename
  --n_pts n_pts         number of scatter points to sample
  --interpolant_type interpolant_type
                        type of interpolant. shepard,hardy,local_hardy, or
                        local_shepard
  --resolution resolution
                        resolution of uniform grid
  --r_sq r_sq           r_sq for hardy. -1 is mean of all distances. 0 is min.
                        1 is max. inbetween values linearly interpolate
                        between min and max
  --num_k num_k         num k for local
  --colormap colormap   color map
  --idx_slice idx_slice
                        index for slicing in each dimension. defaults to
                        middle slice

*****

Once running use 
compare - to compare against same slice in ground truth volume
difference - to create difference image against ground truth (must have same resolution)
slice - to change the slice index
r - to change the r_sq interpolation value
