import matplotlib.pyplot as plt
import numpy as np
# import data_loader
# import visualizer
# import argparse
# import sys
import scipy.spatial
import util

class Vor_Wrapper():
    def __init__(self,pts,f_vals, bbox = [-1,2,-1,2]):
        vor = scipy.spatial.Voronoi(pts)
        vor.close()
        self.vor = vor
        self.points = np.array(vor.points) 
        self.f_vals = f_vals

        # self.point_region = np.array(vor.point_region)

        self.regions, self.vertices, self.ridge_vertices, self.ridge_points = self.voronoi_finite_polygons_2d()

        print len(self.regions)
        self.point_region = np.array(range(len(self.regions)))
        # print self.point_region

        # self.vertices = np.array(vor.vertices)
        # self.ridge_points = np.array(vor.ridge_points)
        # self.ridge_vertices = np.array(vor.ridge_vertices)
        
        
        assert len(self.ridge_points.shape)==2 and self.ridge_points.shape[1]==2
        # self.regions = vor.regions 
        self.bbox = bbox
        


        # self.region_polys = []
        # for idx in range(len(self.regions)):
        #     if len(self.regions[idx])==0:
        #         self.region_polys.append([])
        #         continue
        #     # print self.get_region_poly(idx)
        #     self.region_polys.append(self.get_region_poly(idx))
            # if len(self.region_polys[-1])<0:
            # print 'IDX',idx
            # print self.region_polys[-1]
        

    def voronoi_finite_polygons_2d(self):
        # modified from this stack overflow answer:
        # https://stackoverflow.com/questions/36063533/clipping-a-voronoi-diagram-python

        """
        Reconstruct infinite voronoi regions in a 2D diagram to finite
        regions.
        Parameters
        ----------
        vor : Voronoi
            Input diagram
        radius : float, optional
            Distance to 'points at infinity'.
        Returns
        -------
        regions : list of tuples
            Indices of vertices in each revised Voronoi regions.
        vertices : list of tuples
            Coordinates for revised Voronoi vertices. Same as coordinates
            of input vertices, with 'points at infinity' appended to the
            end.
        """
        vor = self.vor
        if vor.points.shape[1] != 2:
            raise ValueError("Requires 2D input")

        new_regions = []
        new_vertices = vor.vertices.tolist()
        new_ridge_points = np.array(vor.ridge_points)
        new_ridge_vertices = np.array(vor.ridge_vertices)
        new_point_region = []

        center = vor.points.mean(axis=0)
        # if radius is None:
        radius = vor.points.ptp().max()*2

        # Construct a map containing all ridges for a given point
        all_ridges = {}
        for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
            all_ridges.setdefault(p1, []).append((p2, v1, v2))
            all_ridges.setdefault(p2, []).append((p1, v1, v2))

        visited = {}

        # Reconstruct infinite regions
        for p1, region in enumerate(vor.point_region):
            vertices = vor.regions[region]

            if all(v >= 0 for v in vertices):
                # finite region
                new_regions.append(vertices)
                continue

            # reconstruct a non-finite region
            ridges = all_ridges[p1]
            new_region = [v for v in vertices if v >= 0]

            for p2, v1, v2 in ridges:
                if v2 < 0:
                    v1, v2 = v2, v1
                if v1 >= 0:
                    # finite ridge: already in the region
                    continue

                key = (p1,p2) if p1<p2 else (p2,p1)
                if key in visited:
                    new_region.append(visited[key])
                else:

                    # Compute the missing endpoint of an infinite ridge

                    t = vor.points[p2] - vor.points[p1] # tangent
                    t /= np.linalg.norm(t)
                    n = np.array([-t[1], t[0]])  # normal

                    midpoint = vor.points[[p1, p2]].mean(axis=0)
                    direction = np.sign(np.dot(midpoint - center, n)) * n
                    far_point = vor.vertices[v2] + direction * radius

                    
                    new_vertices.append(far_point.tolist())
                    idx_new_vertex = len(new_vertices)-1
                    
                    new_region.append(idx_new_vertex)

                    # find ridge p1 p2
                    # print p1, p2
                    idx_r = np.logical_or(new_ridge_points==p1,new_ridge_points==p2)
                    # print new_ridge_points
                    

                    idx_r = np.sum(idx_r,axis = 1)==2
                    # print idx_r
                    assert np.sum(idx_r)==1
                    idx_r = np.where(idx_r)[0][0]
                    # print idx_r
                    # print new_ridge_vertices
                    dim = 0 if new_ridge_vertices[idx_r,0]==-1 else 1
                    assert new_ridge_vertices[idx_r,dim]==-1
                    new_ridge_vertices[idx_r,dim]=idx_new_vertex
                    # print new_ridge_vertices
                    visited[key]=idx_new_vertex
                    # raw_input()

            # sort region counterclockwise
            vs = np.asarray([new_vertices[v] for v in new_region])
            c = vs.mean(axis=0)
            angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
            new_region = np.array(new_region)[np.argsort(angles)]

            # finish
            new_regions.append(new_region.tolist())
            new_point_region.append(p1)

        return new_regions, np.asarray(new_vertices), new_ridge_vertices, new_ridge_points


    def show_simple(self, figure = None, scalarMap = None):
        plt.ion()

        if figure is None:    
            plt.figure()
        # else:
        #     plt.figure(figure.number)
        
        for region in self.regions:
            polygon = self.vertices[region]
            polygon = np.array(list(polygon)+[polygon[0]])
            plt.plot(polygon[:,0],polygon[:,1],'-k',linewidth = 1)
        
        
        if scalarMap is not None:
            plt.scatter(self.points[:, 0], self.points[:, 1], c = scalarMap.to_rgba(self.f_vals),marker = 'o', s = 25)
        else:
            plt.plot(self.points[:, 0], self.points[:, 1], c = 'ko')

        plt.plot(self.vertices[:, 0], self.vertices[:, 1], 'ko')


        # plt.axis('equal')
        # plt.xlim(self.bbox[0] - 0.01, self.bbox[1] + 0.01)
        # plt.ylim(self.bbox[2] - 0.01, self.bbox[3] + 0.01)
        # plt.ylim(vor.min_bound[1] - 0.1, vor.max_bound[1] + 0.1)

        # plt.savefig('voro.png')
        if figure is None:
            plt.show()
        # scipy.spatial.voronoi_plot_2d(self.vor)


    def show_simple_vor(self):
        scipy.spatial.voronoi_plot_2d(self.vor)

    def get_region_poly(self,region_idx):
        # region = self.regions[region_idx]

        pt_idx = np.where(self.point_region==region_idx)[0][0]
        # print pt_idx

        rel_ridge_bin = np.sum(self.ridge_points==pt_idx,axis = 1)>0
        ridge_vertices = self.ridge_vertices[rel_ridge_bin,:]
        ridge_points = self.ridge_points[rel_ridge_bin,:]

        # if region_idx == 6:
        #     print self.ridge_points
        #     print self.point_region
        #     print pt_idx
        #     print np.sum(self.ridge_points==pt_idx,axis = 1)
        #     print rel_ridge_bin
        #     print ridge_vertices
        #     print ridge_points
        # ridge_end_points = []

        # print 'look here',ridge_vertices
        if len(ridge_vertices)==2:
            ridge_end_points = self.choose_tri_endpoints(ridge_vertices,ridge_points,pt_idx)

            return ridge_end_points

        ridge_end_points = []
        for idx_ridge,ridge_vertex in enumerate(ridge_vertices):
            ridge_point = ridge_points[idx_ridge,:]
            # print ridge_point
            # print ridge_vertex
            
            if np.any(ridge_vertex==-1):
                pts_int = self.get_border_pts(self.points[ridge_point,:])
                # non infinite vertex
                v_good = ridge_vertex[ridge_vertex>-1]

                # non infinite ridge on that vertex
                idx_good = np.logical_or(ridge_vertices==v_good,ridge_vertices!=-1)
                idx_good = np.where(np.sum(idx_good,axis = 1)==2)[0]
                # print len(idx_good)

                if len(idx_good)==1:
                    idx_good = idx_good[0]
                    p1 = self.vertices[ridge_vertices[idx_good,0],:]
                    p2 = self.vertices[ridge_vertices[idx_good,1],:]
                    b1 = 1 if util.half_plane(self.points[pt_idx],p1,p2)<0 else 0
                    b2 = 1 if util.half_plane(pts_int[0],p1,p2)<0 else 0
                    b3 = 1 if util.half_plane(pts_int[1],p1,p2)<0 else 0
                    assert b2!=b3

                    chosen_pt = pts_int[0] if b1==b2 else pts_int[1]
                    ridge_end_points.append(np.concatenate([chosen_pt[np.newaxis,:],self.vertices[v_good]]))
            else:
                ridge_end_points.append(self.vertices[ridge_vertex,:])

        return ridge_end_points


    def choose_tri_endpoints(self,ridge_vertices,ridge_points, pt_idx):
        # ridge_end_points, v_good, pt_idx):
        v_good = np.unique(ridge_vertices)
        assert v_good.size == 2
        v_good = v_good[v_good>-1]
        assert v_good.size == 1
        v_good = self.vertices[v_good[0],:]

        pts_1 = self.get_border_pts(self.points[ridge_points[0],:])
        pts_2 = self.get_border_pts(self.points[ridge_points[1],:])

        pt = self.points[pt_idx,:]
        chosen_pts = []
        for pt1 in pts_1:
            for pt2 in pts_2:

                tri_pts = np.array([v_good,pt1,pt2])
                conv = scipy.spatial.ConvexHull(tri_pts)
                conv.close()
                tri_pts = conv.points[conv.vertices]

                if util.in_triangle(pt,tri_pts):
                    chosen_pts = [np.concatenate([pt1[np.newaxis,:],v_good[np.newaxis,:]],axis = 0),np.concatenate([v_good[np.newaxis,:],pt2[np.newaxis,:]],axis =0)]
                    break

                # b1 = 1 if util.half_plane(pt,pt1,v_good)<0 else 0
                # b2 = 1 if util.half_plane(pt,v_good,pt2)<0 else 0
                # # b3 = 1 if half_plane(pt,pt2, pt1)<0 else 0
                # # print b1, b2, b3
                # if b1==b2:
                #     chosen_pts = [np.concatenate([pt1[np.newaxis,:],v_good[np.newaxis,:]],axis = 0),np.concatenate([v_good[np.newaxis,:],pt2[np.newaxis,:]],axis =0)]
                #     break
            if len(chosen_pts)>0:
                break

        return chosen_pts


    def get_border_pts(self, p_rel):
        dims = [1,1,0,0]
        bbox = self.bbox
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



    


    def in_convex_poly(self, pt, region_idx,show = False):
        # print self.point_region, region_idx
        pt_idx = np.where(self.point_region==region_idx)[0][0]
        centriod = self.points[pt_idx]

        # rel_ridge_bin = np.sum(self.ridge_points==pt_idx,axis = 1)>0
        # ridge_vertices = self.ridge_vertices[rel_ridge_bin,:]
        # ridge_points = self.ridge_points[rel_ridge_bin,:]

        region = self.regions[region_idx]
        assert len(region)>0
        
        conv_pts = self.vertices[region,:]
        conv_pts = util.make_counter_clockwise(conv_pts)

        return util.in_convex_poly(pt,conv_pts,centriod, show)


    def find_pb(self,pt,region_idx, show = False):
        pt_idx = np.where(self.point_region==region_idx)[0][0]
        centriod = self.points[pt_idx]
        pb = util.get_pb(pt,centriod)

        

        if show:
            region = self.regions[region_idx]
            conv_pts = self.vertices[region,:]
            conv_pts = util.make_counter_clockwise(conv_pts)
            conv_pts = np.array(list(conv_pts)+[conv_pts[0]])

            conv_pts = np.array(conv_pts)
            plt.ion()
            plt.figure()
            plt.plot(conv_pts[:,0],conv_pts[:,1],'-b')
            plt.plot(pt[0],pt[1],'*r')
            plt.plot(centriod[0],centriod[1],'*g')
            plt.plot(pb[:,0],pb[:,1],'-g')
            plt.plot(pb[:,0],pb[:,1],'-g')
            # plt.plot(ints[:,0],ints[:,1],'or')

            plt.show()

        return pb


    def get_intersection_points(self, pt, region_idx, show = False):
        # assert self.in_convex_poly(pt,region_idx)
        
        pb = self.find_pb(pt,region_idx)

        region = self.regions[region_idx]
        conv_pts = self.vertices[region,:]
        # print region
        conv_pts, v_idx = util.make_counter_clockwise(conv_pts,np.array(region))
        conv_pts = np.array(list(conv_pts)+[conv_pts[0]])
        v_idx = np.array(list(v_idx)+[v_idx[0]])
        # print v_idx
        # print self.vertices.shape
        # raw_input()
        ints = []
        v_ints = []
        for idx_curr in range(conv_pts.shape[0]-1):
            b1 = conv_pts[idx_curr]
            b2 = conv_pts[idx_curr+1]
            p_int = util.get_intersect(pb[0],pb[1],b1,b2)
            if util.on_line_segment(p_int, b1, b2):
                ints.append(p_int)
                v_ints.append(v_idx[[idx_curr,idx_curr+1]])

        ints = np.array(ints)
        v_ints = np.array(v_ints)

        assert ints.shape[0]==2

        # print v_ints.shape
        
        if show:
            plt.ion()
            plt.gcf()
            plt.plot(conv_pts[:,0],conv_pts[:,1],'-b')
            plt.plot(pt[0],pt[1],'*r')
            plt.plot(pb[:,0],pb[:,1],'-g')
            plt.plot(pb[:,0],pb[:,1],'-g')
            plt.plot(ints[:,0],ints[:,1],'or')
            plt.plot(self.vertices[v_ints[0],0],self.vertices[v_ints[0],1],'oc')
            plt.plot(self.vertices[v_ints[1],0],self.vertices[v_ints[1],1],'om')

            plt.show()

        return ints, v_ints

    def get_neighbor(self, region_idx, v_int):
        # print v_int
        # print self.ridge_vertices
        # print self.ridge_points

        old_pt_idx = np.where(self.point_region==region_idx)[0][0]
        
        ridge_idx = np.logical_and(self.ridge_vertices[:,0]==v_int[0],self.ridge_vertices[:,1]==v_int[1])+np.logical_and(self.ridge_vertices[:,0]==v_int[1],self.ridge_vertices[:,1]==v_int[0])
        ridge_idx = np.where(ridge_idx>0)[0]
        if ridge_idx.size==0:
            return -1
        else:
            assert ridge_idx.size==1

            ridge_pt = self.ridge_points[ridge_idx][0]
            assert np.sum(ridge_pt==old_pt_idx)>0
            new_pt_idx = ridge_pt[0] if ridge_pt[1]==old_pt_idx else ridge_pt[1]
            new_region_idx = self.point_region[new_pt_idx]
            return new_region_idx

    def get_area(self,pt,ints,region_idx, show = False):
        pt_hp = util.half_plane(pt,ints[0],ints[1])
        vertices = self.vertices[self.regions[region_idx],:]
        vertices_hp = [util.half_plane(vertex,ints[0],ints[1]) for vertex in vertices]
        vertices_hp = np.array(vertices_hp) 
        vertices_keep = vertices[vertices_hp*pt_hp>0,:]
        vertices_keep = np.vstack([vertices_keep,ints])
        # print vertices_keep.shape
        area = util.get_area(vertices_keep)
        # conv = scipy.spatial.ConvexHull(vertices_keep)
        # conv.close()

        if show:
            # plt.ion()
            plt.gcf()
            plt.plot(vertices_keep[:,0],vertices_keep[:,1],'-r')
            plt.show()

        return area


    def interpolate_sib(self,pts):
        f_vals = []
        for pt in pts:
            f_vals.append(self.sib_pt(pt))
        return f_vals

    def interpolate_nearest(self,pts):
        f_vals = []
        for pt in pts:
            f_vals.append(self.nearest_pt(pt))
        return f_vals



    def nearest_pt(self,pt):
        region_idx = -1
        for region_idx_curr in range(len(self.regions)):
            if self.in_convex_poly(pt,region_idx_curr):
                region_idx = region_idx_curr
                break
        
        if region_idx<0:
            print 'pt outside voronoi!'
            return -1
        pt_idx = np.where(self.point_region==region_idx)[0][0]
        return self.f_vals[pt_idx]

    def sib_pt(self,pt):
        # find region it belongs to 
        region_idx = -1
        for region_idx_curr in range(len(self.regions)):
            if self.in_convex_poly(pt,region_idx_curr):
                region_idx = region_idx_curr
                break
        
        if region_idx<0:
            print 'pt outside voronoi!'
            return -1

        # print region_idx
        visited = [-1]
        areas = []
        f_vals = []
        while True:
            pt_idx = np.where(self.point_region==region_idx)[0][0]
            ints, v_ints = self.get_intersection_points(pt, region_idx, show = False)

            visited.append(region_idx)
            areas.append(self.get_area(pt, ints, region_idx))
            f_vals.append(self.f_vals[pt_idx])
            # raw_input()
            neighbors = [self.get_neighbor(region_idx,v_int) for v_int in v_ints]
            # print neighbors
            region_idx = [r for r in neighbors if r not in visited]
            # print region_idx
            if len(region_idx)==0:
                break
            
            region_idx = region_idx[0]

        # print areas
        # print f_vals
        f_vals = np.array(f_vals)
        areas = np.array(areas)
        sib_val = np.sum(f_vals*areas)/np.sum(areas)

        return sib_val


        # return neighbor region

        






