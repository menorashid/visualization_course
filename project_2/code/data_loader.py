
import numpy as np
import visualizer
import os
import matplotlib.pyplot as plt
def generate_samples(n_pts):
    pts = np.random.random((n_pts,3))
    # pts = pts * 2 -1
    return pts

def sphere(pts):
    f_val = np.sum(pts**2,1)
    min_val = 0
    max_val = np.sum(np.array([1,1,1])**2)
    return f_val,min_val,max_val

def get_scattered_pts(scatter_fun,n_pts):
    
    if type(n_pts)==int:
        pts = generate_samples(n_pts)
    else:
        pts = n_pts

    vol = None
    if scatter_fun[0]=='sphere':
        f_vals, min_val, max_val = sphere(pts)
    elif scatter_fun[0].endswith('.raw') and os.path.exists(scatter_fun[0]):
        if len(scatter_fun)<2:
            raise ValueError('Need volume size for file "%s".' % scatter_fun[0])    
        
        vol = np.fromfile(scatter_fun[0],dtype=np.uint8)
        
        scatter_fun[1] = int(scatter_fun[1])
        vol.shape = (scatter_fun[1],scatter_fun[1],scatter_fun[1])
        
        vol = vol.astype(float)
        vol = vol-np.min(vol)
        vol = vol/np.max(vol)
        min_val = np.min(vol)
        max_val = np.max(vol)

        # vol_img_xy = vol[:,:,32]
        #display it using greyscale color map
        # plt.imshow(vol_img_xy,cmap="gray",interpolation='nearest')
        # pts_to_int_vec = range(0,scatter_fun[1],10)
        # pts = np.meshgrid(pts_to_int_vec,pts_to_int_vec,pts_to_int_vec);
        # pts = np.concatenate([arr.flatten()[:,np.newaxis] for arr in pts],1)
        # pts = pts.astype(int)
        # print pts.shape
        # print pts[:10]
        pts = np.random.randint(scatter_fun[1],size=(n_pts,3))




        f_vals = np.array([vol[tuple(pt)] for pt in pts]) 

        # bring pts to unit volume
        pts = pts.astype(float)
        pts = pts/float(scatter_fun[1])

        print min_val,max_val
    else:
        raise ValueError('Scatter Function/File "%s" not defined.' % scatter_fun)
        
    return pts, f_vals, min_val, max_val, vol



def main():
    pts = generate_samples(100)
    f_vals,min_val,max_val = sphere(pts)

    visualizer.scatter_plot(pts, f_vals,colormap='cool',  min_val = min_val, max_val = max_val)
    raw_input()



if __name__=='__main__':
    main()

    

