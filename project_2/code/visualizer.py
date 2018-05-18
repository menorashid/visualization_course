from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import numpy as np

def get_scaled_colormap(colormap, min_val, max_val):
    cmap = plt.get_cmap(colormap)
    cNorm = colors.Normalize(vmin=min_val,vmax = max_val)
    scalarMap = cmx.ScalarMappable(norm = cNorm, cmap = cmap)
    return scalarMap


def surface_plot(pts, f_vals, pts_int, f_vals_int, resolution, colormap = 'Greys', min_val = 0., max_val = 1.,alpha = 0.3,marker = '^'):
    
    scalarMap = get_scaled_colormap(colormap,min_val,max_val)
    # f_vals = 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # X, Y = np.meshgrid(np.linspace(0.,1.,resolution), np.linspace(0.,1.,resolution))

    # Y = np.reshape(pts[:,1],(resolution,resolution));
    # X = np.reshape(pts[:,0],(resolution,resolution));

    # c_vals = [scalarMap.to_rgba(f_val) for f_val in f_vals_int]
    # ax.plot(pts_int[:,0], pts_int[:,1], pts_int[:,2], c = c_vals, marker = marker)

    for idx_pt_curr,pt_curr in enumerate(pts_int):
        ax.scatter(pt_curr[0], pt_curr[1], pt_curr[2], c=scalarMap.to_rgba(f_vals_int[idx_pt_curr]),marker = marker)

    
    for idx_pt_curr,pt_curr in enumerate(pts):
        ax.scatter(pt_curr[0], pt_curr[1], pt_curr[2], c=scalarMap.to_rgba(f_vals[idx_pt_curr]),marker = 'o')

    plt.ion()
    # rotate the axes and update
    # for angle in range(0, 360):
    ax.view_init(30, 120)
    plt.draw()
    plt.show()
     
def plot_slice(fval_slice,title = '',colormap = 'Greys', min_val = 0., max_val = 1.):
    scalarMap = get_scaled_colormap(colormap,min_val,max_val)
    im_3d = np.zeros((fval_slice.shape[0],fval_slice.shape[1],3))
    for r in range(fval_slice.shape[0]):
        for c in range(fval_slice.shape[1]):
            im_3d[r,c,:]=scalarMap.to_rgba(fval_slice[r,c])[:3]

    plt.figure();
    plt.ion()
    plt.title(title)
    plt.imshow(im_3d,interpolation='nearest')
    plt.show()

def display_slices(f_vals_int,idx_slice,title_pre, colormap = 'Greys', min_val = 0., max_val = 1., special = False):
    plt.figure()
    plt.ion()
    plt.set_cmap(colormap)
    
    for dim in range(3):
        plt.subplot(130+dim+1)
        fval_slice = np.take(f_vals_int,indices =[idx_slice],axis= dim)
        fval_slice = fval_slice.squeeze()
        title_curr = title_pre+'Dim %d, Index %d' % (dim, idx_slice)

        if special:
            min_val = np.min(fval_slice)
            max_val = np.max(fval_slice)

        scalarMap = get_scaled_colormap(colormap,min_val,max_val)
        im_3d = np.zeros((fval_slice.shape[0],fval_slice.shape[1],3))
        for r in range(fval_slice.shape[0]):
            for c in range(fval_slice.shape[1]):
                im_3d[r,c,:]=scalarMap.to_rgba(fval_slice[r,c])[:3]

        plt.title(title_curr)
        im_handle = plt.imshow(im_3d,interpolation='nearest')
        plt.colorbar(im_handle,orientation='horizontal')
        plt.show()




def scatter_plot(pts, f_vals, colormap = 'Greys', min_val = 0., max_val = 1.,marker = 'o'):
    scalarMap = get_scaled_colormap(colormap,min_val,max_val)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # load some test data for demonstration and plot a wireframe
    # X, Y, Z = axes3d.get_test_data(0.1)
    for idx_pt_curr,pt_curr in enumerate(pts):
        ax.scatter(pt_curr[0], pt_curr[1], pt_curr[2], c=scalarMap.to_rgba(f_vals[idx_pt_curr]),marker = marker)

    plt.ion()
    # rotate the axes and update
    # for angle in range(0, 360):
    ax.view_init(30, 120)
    plt.draw()
    plt.show()
    #     plt.pause(.001)


