import sys
import numpy as np
import data_loader
import visualizer
import scipy.spatial
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
import argparse

def shepard_slow(pts_to_int,pts,f_vals, distance_thresh=1e-9):
    
    print 'pts_to_int', pts_to_int.shape
    print 'pts', pts.shape
    print 'f_vals', f_vals.shape
    
    f_out = np.zeros((pts_to_int.shape[0],))
    breaking = False

    for r in range(pts_to_int.shape[0]):
        inv_total = 0
        for c in range(pts.shape[0]):
            d = np.sum((pts_to_int[r]-pts[c])**2)
            # if d<0.25:
            #   continue

            if d<distance_thresh:
                f_out[r] = f_vals[c]
                breaking = True
                break
            
            f_out[r] = f_out[r] + (1./d * f_vals[c])
            inv_total = inv_total + 1./d

        if not breaking:
            f_out[r] = f_out[r]/inv_total
        else:
            breaking = False

    return f_out

# def hardy(pts_to_int, pts, f_vals, r_sq = None, num_k = None):

#     # remove duplicate pts
#     unique_index = np.unique(pts.dot(np.random.rand(3)), return_index= True)[1]
#     pts_unique = pts[unique_index]
#     f_vals = f_vals[unique_index]
#     print pts_unique.shape
#     print pts.shape


#     distance_self = scipy.spatial.distance.cdist(pts,pts,'euclidean')**2
#     # print np.all(distance_self == distance_self.T)
#     if r_sq is None:
#         r_sq =np.mean(distance_self)
#         # 128
#         # np.sum(np.triu(distance_self))/float(distance_self.shape[0]*(distance_self.shape[0]-1))
#         # 0.5*(np.max(distance_self)-  np.min(distance_self))
#         # 
#         # np.max(distance_self)
#         # 

#     print r_sq
#     # r_sq = 0.00000001

#     h_self = np.sqrt(r_sq+distance_self).T
#     # print h_self.shape

#     # det = np.linalg.det(h_self)
#     # print 'det',det
#     # print np.where(distance_self==0)

#     # new_array = [' '.join([str(val) for val in row]) for row in pts]
#     # uniques = np.unique(new_array)
#     # print uniques.shape
#     # # print pts[:10]
#     # # raw_input()
#     # print np.min(f_vals),np.max(f_vals),np.min(h_self),np.max(h_self),h_self.shape

#     # if num_k is None:
    
#     # coeff = np.matmul(np.linalg.inv(h_self),f_vals[:,np.newaxis])
#     # print np.all(h_self == h_self.T)
#     coeff,_,_,_ = np.linalg.lstsq(h_self, f_vals[:,np.newaxis])
#     # np.linalg.solve(h_self.T, f_vals[:,np.newaxis])
#     # np.linalg.solve(h_self, f_vals[:,np.newaxis])
#     # coeff,_,_,_ = np.linalg.lstsq(h_self, f_vals[:,np.newaxis])
#     # coeff = coeff.T
#     # coeff = np.tile(coeff,(pts_to_int.shape[0],1))


#     # h_mat_slow = np.zeros((pts.shape[0],pts.shape[0]))
#     # for r in range(h_mat_slow.shape[0]):
#     #     for c in range(h_mat_slow.shape[1]):
#     #         d = np.sum((pts[r]-pts[c])**2)
#     #         h_mat_slow[r,c] = np.sqrt(r_sq+d)
#     # coeff = np.linalg.solve(h_mat_slow, f_vals[:,np.newaxis])
#     f_vals_rec = np.matmul(h_self,coeff)
#     print f_vals_rec.shape
#     print np.min(f_vals - f_vals_rec),np.max(f_vals - f_vals_rec)

#     # raw_input()
#     # coeff = coeff.T
#     print coeff.shape
#     # coeff = np.tile(coeff,(pts_to_int.shape[0],1))



#     if num_k is None:
#         print 'NONE'
#         distance = scipy.spatial.distance.cdist(pts_to_int,pts,'euclidean')**2
#     else:
#         kdt = KDTree(pts, leaf_size=30, metric='euclidean')
#         distance, sort_idx = kdt.query(pts_to_int, num_k)
#         distance = distance**2
#         coeff =  np.array([coeff[idx_idx,idx_select] for idx_idx, idx_select in enumerate(sort_idx)])


#     h = np.sqrt(r_sq+distance)
#     h = coeff*h
#     h = np.sum(h,1)
    

#     return h

def get_hardy_coeff(pts,f_vals,r_sq):
    distance_self = scipy.spatial.distance.cdist(pts,pts,'euclidean')**2

    # print 'r_sq',r_sq
    
    h_self = np.sqrt(r_sq+distance_self)
    inv = np.linalg.inv(h_self)
    
    # det = np.linalg.det(h_self)
    # print 'det',det
    coeff = np.dot(inv,f_vals[:,np.newaxis])
    # print coeff
    # print  np.allclose(np.dot(h_self,coeff),f_vals[:,np.newaxis])
    return coeff


def hardy(pts_to_int, pts, f_vals, r_sq_type = -1, num_k = None):

    # remove duplicate pts
    print pts.shape
    unique_index = np.unique(pts.dot(np.random.rand(3)), return_index= True)[1]
    pts = pts[unique_index]
    f_vals = f_vals[unique_index]
    print 'unique',pts.shape,f_vals.shape, np.min(f_vals),np.max(f_vals)


    p_wise_distances = np.array(scipy.spatial.distance.pdist(pts,'euclidean')**2)
    print 'p_wise_distances',p_wise_distances.shape
    
    if r_sq_type<0:
        r_sq = np.mean(p_wise_distances)
        print r_sq
    else:
        assert 0.<=r_sq_type<=1.
        r_sq = r_sq_type*np.max(p_wise_distances)+(1-r_sq_type)*np.min(p_wise_distances)
        print r_sq.shape
        

    

    if num_k is None:
        coeff = get_hardy_coeff(pts,f_vals,r_sq)
        coeff = coeff.T
        print pts.shape,pts_to_int.shape, coeff.shape
        # coeff = np.tile(coeff,(pts_to_int.shape[0],1))
        distance = scipy.spatial.distance.cdist(pts_to_int,pts,'euclidean')**2
        print distance.shape
    else:
        kdt = KDTree(pts, leaf_size=30, metric='euclidean')
        coeff = np.zeros((pts_to_int.shape[0],num_k))
        # r_sq = np.zeros((pts_to_int.shape[0],1))
        distance, sort_idx = kdt.query(pts_to_int, num_k)
        distance = distance**2
        for idx_curr, sort_idx_curr in enumerate(sort_idx):
            coeff_curr = get_hardy_coeff(pts[sort_idx_curr,:],f_vals[sort_idx_curr],r_sq)
            coeff[idx_curr] = coeff_curr.T[0]
            # r_sq[idx_curr] = r_sq_curr
        


    h = np.sqrt(r_sq+distance)
    h = coeff*h
    h = np.sum(h,1)
    
    # print np.min(h),np.max(h)

    return h,r_sq





def shepard(pts_to_int, pts, f_vals, distance_thresh = 1e-5, num_k = None):
    f_vals = np.tile(f_vals[np.newaxis,:],(pts_to_int.shape[0],1))
        
    if num_k is None:
        distance = scipy.spatial.distance.cdist(pts_to_int,pts,'euclidean')**2
        print f_vals.shape
    else:
        kdt = KDTree(pts, leaf_size=30, metric='euclidean')
        distance, sort_idx = kdt.query(pts_to_int, num_k)
        distance = distance**2
        
        f_vals =  np.array([f_vals[idx_idx,idx_select] for idx_idx, idx_select in enumerate(sort_idx)])
    

    too_close = distance<=distance_thresh
    bin_prob = np.sum(too_close,1)>0
    print 'TOO CLOSE OCCURENCES %d out of %d' % (np.sum(bin_prob), pts_to_int.shape[0])
    print np.min(distance),np.max(distance)



    inv_distance = 1./distance
    inv_distance[bin_prob,:]=0.
    inv_distance[too_close]=1.
    
    
    w = inv_distance/np.sum(inv_distance,1,keepdims=True) 
    numo = w * f_vals
    f_out = np.sum(numo,1)
    
    return f_out


def get_interpolated_f_vals(interpolant_type,pts_to_int,pts,f_vals,r_sq, num_k):
    if interpolant_type=='shepard':
        f_vals_int = shepard(pts_to_int,pts,f_vals)
    elif interpolant_type=='shepard_slow':
        f_vals_int = shepard_slow(pts_to_int,pts,f_vals)
    elif interpolant_type=='local_shepard':
        assert num_k is not None and num_k>0
        f_vals_int = shepard(pts_to_int,pts,f_vals,num_k = num_k)
    elif interpolant_type=='hardy':
        f_vals_int,_ = hardy(pts_to_int,pts,f_vals,r_sq_type=r_sq)
    elif interpolant_type=='local_hardy':
        f_vals_int,_ = hardy(pts_to_int,pts,f_vals,r_sq_type=r_sq,num_k = num_k)
    else:
        raise ValueError('Interpolant type "%s" not defined.' % interpolant_type)

    return f_vals_int



def interpolate_and_plot(scatter_fun,n_pts,interpolant_type,resolution,idx_slice = None, r_sq = None , num_k = 10,colormap = 'jet', diff = False):

    pts, f_vals, min_val, max_val = data_loader.get_scattered_pts(scatter_fun,n_pts)
    print min_val,max_val,np.min(f_vals),np.max(f_vals),pts.shape
    
    pts_to_int_vec = np.linspace(0.,1.,resolution)
    pts_to_int = np.meshgrid(pts_to_int_vec,pts_to_int_vec,pts_to_int_vec);
    pts_to_int = np.concatenate([arr.flatten()[:,np.newaxis] for arr in pts_to_int],1)

    if idx_slice is None:
        idx_slice = resolution//2
    
    # testers
    if scatter_fun[0]=='sphere':
        f_vals_int_gt,_,_ = data_loader.sphere(pts_to_int)
        f_vals_int_gt = np.reshape(f_vals_int_gt,(resolution,resolution,resolution))
        for dim in range(3):
            slice_curr = np.take(f_vals_int_gt,indices =[idx_slice],axis= dim)
            slice_curr = slice_curr.squeeze()
            title_curr = 'GT Dim %d, Index %d' % (dim, idx_slice)
            visualizer.plot_slice(slice_curr,title=title_curr, colormap = colormap, min_val = min_val, max_val = max_val)
                # min_val = np.min(f_vals_int_gt), max_val = np.max(f_vals_int_gt))

    
    f_vals_int = get_interpolated_f_vals(interpolant_type,pts_to_int,pts,f_vals,r_sq, num_k)
    f_vals_int = np.reshape(f_vals_int,(resolution,resolution,resolution))
    print np.min(f_vals_int),np.max(f_vals_int),pts_to_int.shape

    for dim in range(3):
        slice_curr = np.take(f_vals_int,indices =[idx_slice],axis= dim)
        slice_curr = slice_curr.squeeze()
        title_curr = 'Dim %d, Index %d' % (dim, idx_slice)
        visualizer.plot_slice(slice_curr,title=title_curr, colormap = colormap, min_val = min_val, max_val = max_val)
            


def main_loop(scatter_fun,n_pts,interpolant_type,resolution,idx_slice = None, r_sq = None , num_k = 10,colormap = 'jet', diff = False):
    pts, f_vals, min_val, max_val,vol = data_loader.get_scattered_pts(scatter_fun,n_pts)
    pts_to_int_vec = np.linspace(0.,1.,resolution)
    
    pts_to_int = []
    for x in pts_to_int_vec:
        for y in pts_to_int_vec:
            for z in pts_to_int_vec:
                pts_to_int.append([x,y,z])
    pts_to_int = np.array(pts_to_int)

    if scatter_fun[0]=='sphere':
        vol,_,_ = data_loader.sphere(pts_to_int)
        vol = np.reshape(vol,(resolution,resolution,resolution))

    if idx_slice is None:
        idx_slice = resolution//2


    f_vals_int = get_interpolated_f_vals(interpolant_type,pts_to_int,pts,f_vals,r_sq, num_k)
    print f_vals_int.shape
    f_vals_int = np.reshape(f_vals_int,(resolution,resolution,resolution))
    print np.min(f_vals_int),np.max(f_vals_int),pts_to_int.shape

    visualizer.display_slices(f_vals_int,idx_slice,interpolant_type+'\n', colormap, min_val = min_val, max_val = max_val)
        # min_val = np.min(f_vals_int), max_val = np.max(f_vals_int))
        # special = True)
        # min_val = min_val, max_val = max_val)

    while (True):
        idx_slice_gt = int(idx_slice/float(resolution)*vol.shape[0])

        to_do = raw_input('Type command:\n')
        if to_do=='compare':
            if vol is None:
                print 'Can not compare'
            else:
                visualizer.display_slices(vol,idx_slice_gt,'GT\n', colormap = colormap, min_val = min_val, max_val = max_val)
                visualizer.display_slices(f_vals_int,idx_slice,interpolant_type+'\n', colormap = colormap, min_val = min_val, max_val = max_val)
        elif to_do=='difference':
            if vol is None:
                print 'Can not compare'
            else:
                diff = np.abs(vol-f_vals_int)
                print 'Min diff:',np.min(diff),'Max diff', np.max(diff)
                visualizer.display_slices(vol,idx_slice_gt,'GT\n', colormap = colormap, min_val = min_val, max_val = max_val)
                visualizer.display_slices(diff,idx_slice,'Diff\n', colormap = colormap, special = True)
        elif to_do=='slice':
            idx_slice = int(raw_input('slice_idx:\n'))
            assert resolution>idx_slice>=0
            visualizer.display_slices(f_vals_int,idx_slice,interpolant_type+'\n', colormap = colormap, min_val = min_val, max_val = max_val)
        elif to_do=='r':
            assert interpolant_type=='local_hardy'
            r_sq = float(raw_input('r type :\n'))
            assert r_sq<=1.

            f_vals_int,r_sq_curr = hardy(pts_to_int,pts,f_vals,r_sq_type=r_sq,num_k = num_k)
            f_vals_int = np.reshape(f_vals_int,(resolution,resolution,resolution))
            print np.min(f_vals_int),np.max(f_vals_int),pts_to_int.shape
            title_pre = '%s \nr %.4f\n' % (interpolant_type,r_sq_curr)
            visualizer.display_slices(f_vals_int,idx_slice,title_pre, colormap, min_val = min_val, max_val = max_val)

        else:
            print 'Type "compare", "difference", "slice" or "r"'


def main(args):


    parser = argparse.ArgumentParser(description='Interpolate scattered pts')
    parser.add_argument('--scatter_fun', metavar='scatter_fun', type=str, nargs = '+', default = ['sphere'], help='scatter function or file. enter resolution after filename')
    parser.add_argument('--n_pts', metavar='n_pts', default = 1000, type=int, help='number of scatter points to sample')
    parser.add_argument('--interpolant_type', metavar='interpolant_type',default = 'shepard', type=str, help='type of interpolant. shepard,hardy,local_hardy, or local_shepard')
    parser.add_argument('--resolution', metavar='resolution',default = 10, type=int, help='resolution of uniform grid')
    parser.add_argument('--r_sq', metavar='r_sq',default = 0., type=int, help='r_sq for hardy. -1 is mean of all distances. 0 is min. 1 is max. inbetween values linearly interpolate between min and max')
    parser.add_argument('--num_k', metavar='num_k',default = 10, type=int, help='num k for local')
    parser.add_argument('--colormap', metavar='colormap',default = 'jet', type=str, help='color map')
    parser.add_argument('--idx_slice', metavar='idx_slice',default = None, type=int, help='index for slicing in each dimension. defaults to middle slice')

    args = parser.parse_args(args[1:])
    args = vars(args)
    print args
    main_loop(**args)


if __name__=='__main__':
    main(sys.argv)