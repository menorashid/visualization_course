
# def half_plane ( p1,  p2,  p3):
#     return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])





def in_triangle(pt, tri_pts):
    b1 = 1 if half_plane(pt, tri_pts[0], tri_pts[1]) < 0 else 0
    b2 = 1 if half_plane(pt, tri_pts[1], tri_pts[2]) < 0 else 0
    b3 = 1 if half_plane(pt, tri_pts[2], tri_pts[0]) < 0 else 0
    
    return b1==b2==b3

def in_convex_poly(pt, conv_pts):
    in_poly = False
    centriod = np.mean(conv_pts,0)
    conv_pts = list(conv_pts)+[conv_pts[0]]
    for idx_pt_1 in range(len(conv_pts)-1):
        tri_pts = [conv_pts[idx_pt_1],conv_pts[idx_pt_1+1],centriod]
        if in_triangle(pt,tri_pts):
            in_poly = True
            break

    return in_poly


def make_finite(points, ridge_points,ridge_vertices,vor_vertices,bbox):
    for idx_ridge in range(ridge_points.shape[0]):
        r_v = ridge_vertices[idx_ridge]
        r_p = ridge_points[idx_ridge]

        if not np.any(r_v<0):
            print 'no problem',r_v,r_p
            continue
        
        print 'no problem',r_v,r_p

def get_border_pts(p_rel,bbox):
    dims = [1,1,0,0]
    #bbox -> x_min,x_max,y_min,y_max
    pts_int = np.zeros((4,2))
    for idx_int, (dim_curr, val_curr) in enumerate(zip(dims,bbox)):
        dim_opp = (dim_curr+1)%2
        
        y1 = p_rel[0,dim_opp]
        x1 = p_rel[0,dim_curr]

        y2 = p_rel[1,dim_opp]
        x2 = p_rel[1,dim_curr]
        
        numo = (val_curr - y2)**2 - (val_curr - y1)**2 - x1**2 + x2**2
        deno = x2 - x1
        val = 0.5 * numo/deno
        pts_int[idx_int,dim_curr]= val
        pts_int[idx_int,dim_opp]= val_curr

    bin_keep = np.logical_and(np.logical_and(bbox[0]<=pts_int[:,0],pts_int[:,0]<=bbox[1]),np.logical_and(bbox[2]<=pts_int[:,1],pts_int[:,1]<=bbox[3]))
    pts_int = pts_int[bin_keep,:]
    return pts_int


def get_pb_pts(points, ridge_points):
    pb_pts = []
    for rp in ridge_points:
        print 'rp',rp
        p_rel = points[rp,:]
        p_rel = np.mean(p_rel,axis = 0,keepdims = True)
        pb_pts.append(p_rel)

    
    pb_pts = np.concatenate(pb_pts,axis = 0)
    print pb_pts.shape

    return pb_pts

def get_uni_poly_tups(region):
    poly_tups = []
    for idx_v_idx in range(len(region)-1):
        tup_curr = [region[idx_v_idx],region[idx_v_idx+1]]
        tup_curr.sort()
        tup_curr = ','.join([str(val) for val in tup_curr])
        poly_tups.append(tup_curr)

    # print poly_tups
    poly_tups = list(set(poly_tups))
    # print poly_tups
    # print poly_tups[0].split(',')
    poly_tups = [[int(val) for val in tup.split(',')] for tup in poly_tups]
    # print poly_tups
    return poly_tups


def get_ridge_list(region, ridge_points, ridge_vertices, pt_idx):
    ridge_vertices = np.array(ridge_vertices)
    ridge_points = np.array(ridge_points)
    
    region_ridge_idx = []
    poly_tups = get_uni_poly_tups(region)


    for v_idx,v_n_idx in poly_tups:
        
        rel_idx = np.sum(np.logical_or(ridge_vertices==v_idx,ridge_vertices==v_n_idx),axis = 1)==2

        rel_idx = np.logical_and(rel_idx,np.sum(ridge_points==pt_idx,axis=1))

        assert  1<=np.sum(rel_idx)<=2
        region_ridge_idx += list(np.where(rel_idx)[0])

    return region_ridge_idx