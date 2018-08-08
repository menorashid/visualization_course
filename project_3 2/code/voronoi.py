import matplotlib.pyplot as plt
import numpy as np
import data_loader
import visualizer
import argparse
import sys
import scipy.spatial
from vor_wrapper import Vor_Wrapper
 # import Voronoi, voronoi_plot_2d
# >>> vor = Voronoi(points)
# Plot it:

# >>>
# >>> import matplotlib.pyplot as plt
# >>> voronoi_plot_2d(vor)
# >>> plt.show()



def get_sib(resolution):
    sib_x = np.linspace(0,1,resolution)
    sib_y = np.linspace(0,1,resolution)
    sib_x,sib_y = np.meshgrid(sib_x, sib_y)
    sib_x = np.reshape(sib_x,(sib_x.size,1))
    sib_y = np.reshape(sib_y,(sib_y.size,1))
    sib = np.concatenate([sib_x,sib_y],axis = 1)
    return sib

def main_loop(scatter_fun, n_pts, resolution=16, colormap = 'jet'):
    np.random.seed(seed=2)
    pts, f_vals, min_val, max_val, _ = data_loader.get_scattered_pts(scatter_fun,n_pts)
    sib = get_sib(resolution)    
    print sib.shape
    bbox = [0,1,0,1]
    # colormap = 'jet'
    # return


    idx_arg_x = np.argsort(pts[:,0])
    idx_arg_y = np.argsort(pts[:,1])
    idx_keep = [idx_arg_x[0],idx_arg_x[-1],idx_arg_y[0],idx_arg_y[-1]]
    # remaining_idx = np.array(range(pts.shape[0]))
    # remaining_idx = np.setdiff1d(remaining_idx,idx_keep)

    while True:
        remaining_idx = np.array(range(pts.shape[0]))
        remaining_idx = np.setdiff1d(remaining_idx,idx_keep)

        pts_border = pts[idx_keep,:]
        f_vals_border = f_vals[idx_keep]
        num_pts = pts_border.shape[0]
        vor = Vor_Wrapper(pts_border,f_vals_border,bbox = bbox)

        plt.figure()
        f_vals_sib = vor.interpolate_sib(sib)
        f_vals_sib = np.reshape(f_vals_sib,(resolution,resolution))
        visualizer.plot_vor_sib(f_vals_sib,vor,bbox,colormap,min_val,max_val,'Sibson '+str(num_pts),fig = plt.subplot(1,3,2))
        
        f_vals_nearest = vor.interpolate_nearest(sib)
        f_vals_nearest = np.reshape(f_vals_nearest,(resolution,resolution))
        visualizer.plot_vor_sib(f_vals_nearest,vor,bbox,colormap,min_val,max_val,'Nearest '+str(num_pts),fig = plt.subplot(1,3,1))
        
        _,f_vals_gt,_,_,_ = data_loader.get_scattered_pts(scatter_fun, sib)
        f_vals_gt = np.reshape(f_vals_gt,(resolution,resolution))
        visualizer.plot_fvals(f_vals_gt, colormap, min_val, max_val, bbox, title = 'GT '+str(num_pts),fig = plt.subplot(1,3,0))
        plt.show()

        print str(num_pts)+' Mean square difference Sibson:',np.mean((f_vals_gt - f_vals_sib)**2)
        print str(num_pts)+' Mean square difference Nearest:',np.mean((f_vals_gt - f_vals_nearest)**2)


        to_do = raw_input('Type command:\n')
        if to_do=='double':
            num_pts = min(num_pts*2, pts.shape[0])
            to_add = num_pts - len(idx_keep)
            idx_keep = list(remaining_idx[:to_add])+ idx_keep
        elif to_do=='add':
            to_add = int(raw_input('number to add:\n'))
            num_pts = min(num_pts+to_add, pts.shape[0])
            idx_keep = list(remaining_idx[:to_add])+ idx_keep
        elif to_do=='resolution':
            resolution = int(raw_input('resolution:\n'))
            sib = get_sib(resolution)
        else:
            print 'Type "double", "add",or "resolution"'


    
    raw_input()


def test_ridge_list(scatter_fun, n_pts):
    np.random.seed(seed=2)
    pts, f_vals, min_val, max_val, _ = data_loader.get_scattered_pts(scatter_fun,n_pts)
    
    idx_arg_x = np.argsort(pts[:,0])
    idx_arg_y = np.argsort(pts[:,1])
    idx_keep = [idx_arg_x[0],idx_arg_x[-1],idx_arg_y[0],idx_arg_y[-1]]
    # range(10)
    # [idx_arg_x[0],idx_arg_x[-1],idx_arg_y[0],idx_arg_y[-1],5,10]
    pts_border = pts[idx_keep,:]
    f_vals_border = f_vals[idx_keep]
    print pts_border
    
    vor = Vor_Wrapper(pts_border)
    plt.ion()
    vor.show_simple()
    
    for idx_region_ridge, region_ridge in enumerate(vor.region_polys):
        if len(region_ridge)==0:
            continue

        vor.show_simple()    

        pt_idx = np.where(vor.point_region==idx_region_ridge)[0][0]
        pt = vor.points[pt_idx]
        plt.plot(pt[0],pt[1],'*r')        
        
        for line_curr in region_ridge:
            plt.plot(line_curr[:,0],line_curr[:,1],'--r')


        plt.show()
    raw_input()


def test_inside_outside():

    tri_pts =[[0,0],[1,0],[1,1],[0,1]]
    tri_pts = np.array(tri_pts)
    print tri_pts.shape
    pt = [0.25,0.25]
    print in_convex_poly(pt,tri_pts)
    pt = [1,1.2]
    print in_convex_poly(pt,tri_pts)


def main(args):


    parser = argparse.ArgumentParser(description='Make Voronoi Diagrams and then Sibson them')
    parser.add_argument('--scatter_fun', metavar='scatter_fun', type=str, nargs = '+', default = ['serrated_sphere'], help='scatter function')
    parser.add_argument('--n_pts', metavar='n_pts', default = 1000, type=int, help='number of scatter points to sample')
    parser.add_argument('--resolution', metavar='resolution', default = 16, type=int, help='resolution')
    parser.add_argument('--colormap', metavar='colormap', default = 'jet', type=str, help='colormap')
    # parser.add_argument('--interpolant_type', metavar='interpolant_type',default = 'shepard', type=str, help='type of interpolant. shepard,hardy,local_hardy, or local_shepard')
    # parser.add_argument('--resolution', metavar='resolution',default = 10, type=int, help='resolution of uniform grid')
    # parser.add_argument('--r_sq', metavar='r_sq',default = 0., type=int, help='r_sq for hardy. -1 is mean of all distances. 0 is min. 1 is max. inbetween values linearly interpolate between min and max')
    # parser.add_argument('--num_k', metavar='num_k',default = 10, type=int, help='num k for local')
    # parser.add_argument('--colormap', metavar='colormap',default = 'jet', type=str, help='color map')
    # parser.add_argument('--idx_slice', metavar='idx_slice',default = None, type=int, help='index for slicing in each dimension. defaults to middle slice')

    args = parser.parse_args(args[1:])
    args = vars(args)
    print args
    # main_loop(**args)
    # test_ridge_list(**args)
    main_loop(**args)


    # plt.ion()
    # plt.figure()
    # plt.imshow(np.random.randint(256, size=(224,224,3)))
    # plt.show()
    
    # print 'hello'

if __name__=='__main__':
    main(sys.argv)