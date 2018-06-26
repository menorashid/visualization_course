import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial



def on_line_segment(q, p, r):


    if (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
        q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])):
       return True
 
    return False

    # d = pt - p1
    # d = d/np.linalg.norm(d)

    # d1 = pt - p1
    # d1 = d1/np.linalg.norm(d1)

    # assert np.allclose(d,d1)

    # m = np.sqrt(np.sum((p1 - pt)**2))
    # m1 = np.sqrt(np.sum((p1 - p2)**2))

    # if m<=m1:
    #     return True
    # else:
    #     return False

def get_intersect(a1, a2, b1, b2):
    """ 
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack([a1,a2,b1,b2])        # s for stacked
    h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
    l1 = np.cross(h[0], h[1])           # get first line
    l2 = np.cross(h[2], h[3])           # get second line
    x, y, z = np.cross(l1, l2)          # point of intersection
    if z == 0:                          # lines are parallel
        return (float('inf'), float('inf'))
    return (x/z, y/z)



def get_pb(p1,p2):
    p1_out = (p1+p2)/2.
    if (p2[1] - p1[1]) ==0:
        p2_out =  p1_out
        p2_out[1] = p2_out[1]+0.1
    else:
        m = (p1[0] - p2[0])/(p2[1] - p1[1])
        p2_out =  np.array([0,p1_out[1] - m*p1_out[0]])
        dir_out = p2_out - p1_out
        p2_out = p1_out + 0.1 * dir_out/np.linalg.norm(dir_out)

    return np.array([p1_out,p2_out])




def half_plane (  p1,  p2,  p3):
    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])



def in_triangle( pt, tri_pts):
    b1 = 1 if half_plane(pt, tri_pts[0], tri_pts[1]) < 0 else 0
    b2 = 1 if half_plane(pt, tri_pts[1], tri_pts[2]) < 0 else 0
    b3 = 1 if half_plane(pt, tri_pts[2], tri_pts[0]) < 0 else 0
    # print pt
    # print tri_pts
    # print b1, b2, b3
    return b1==b2==b3



# def in_triangle(pt, tri_pts):
# b1 = 1 if half_plane(pt, tri_pts[0], tri_pts[1]) < 0 else 0
# b2 = 1 if half_plane(pt, tri_pts[1], tri_pts[2]) < 0 else 0
# b3 = 1 if half_plane(pt, tri_pts[2], tri_pts[0]) < 0 else 0

# return b1==b2==b3

def in_convex_poly(pt, conv_pts, centriod = None, show = False):
    in_poly = False
    if centriod is None:
        centriod = np.mean(conv_pts,0)
        
    conv_pts = list(conv_pts)+[conv_pts[0]]
    for idx_pt_1 in range(len(conv_pts)-1):
        tri_pts = [conv_pts[idx_pt_1],conv_pts[idx_pt_1+1],centriod]
        if in_triangle(pt,tri_pts):
            in_poly = True
            break
    
    if show:
        conv_pts = np.array(conv_pts)
        plt.ion()
        plt.figure()
        plt.plot(conv_pts[:,0],conv_pts[:,1],'-b')
        plt.plot(pt[0],pt[1],'*r')
        plt.plot(centriod[0],centriod[1],'*g')
        plt.show()

    return in_poly

def get_area(pts):
    pts = make_counter_clockwise(pts)
    area = 0.
    pts = list(pts)+[pts[0]]
    for idx_pt in range(len(pts)-1):
        p1 = pts[idx_pt]
        p2  = pts[idx_pt+1]
        area+= p1[0]*p2[1] - p2[0]*p1[1]
    area = 0.5* abs(area)
    return area

# def area(p):
#     return 0.5 * abs(sum(x0*y1 - x1*y0
#                          for ((x0, y0), (x1, y1)) in segments(p)))

# def segments(p):
#     return zip(p, p[1:] + [p[0]])


def make_counter_clockwise(tri_pts,v_idx = None):
    conv = scipy.spatial.ConvexHull(tri_pts)
    conv.close()
    tri_pts = tri_pts[conv.vertices]
    if v_idx is not None:
        v_idx = v_idx[conv.vertices]
        return tri_pts,v_idx
    else:
        return tri_pts
    # [::-1,:]
