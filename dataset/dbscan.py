import _init_paths
import open3d as o3d
from PIL import Image
import os
import os.path
import numpy as np
import random
import numpy.ma as ma
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_samples, silhouette_score
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import cv2
from config import Configuration

class Dbscanner:
    def __init__(self, depths, labels, configuration):
        self.config = configuration
        self.list_depth = depths
        self.list_label = labels
        self.dbscan_eps = self.config.dbscan_eps
        self.dbscan_min_pts = self.config.dbscan_min_points
        self.reset_eps = self.dbscan_eps
        self.reset_min_pts = self.dbscan_min_pts
        self.verbose = self.config.debug

        self.length = len(self.list_depth)

        self.xmap = np.array([[j for i in range(self.config.width)] for j in range(self.config.height)])
        self.ymap = np.array([[i for i in range(self.config.width)] for j in range(self.config.height)])

    def project_3d_mask(self, p3d, intrinsic_matrix=np.array([[320, 0., 320],[0., 320, 240],[0., 0., 1.]])):
        p2d = np.dot(p3d * 1000, intrinsic_matrix.T)
        p2d_3 = p2d[:, 2]
        p2d_3[np.where(p2d_3 < 1e-8)] = 1.0
        p2d[:, 2] = p2d_3
        p2d = np.around((p2d[:, :2] / p2d[:, 2:])).astype(np.int32)
        # p2d = p2d[np.where(p2d[:, 0] < self.config.width)]
        # p2d = p2d[np.where(p2d[:, 0] >= 0)]
        # p2d = p2d[np.where(p2d[:, 1] < self.config.height)]
        # p2d = p2d[np.where(p2d[:, 1] >= 0)]
        p2d[:, [1, 0]] = p2d[:, [0, 1]]
        mask = np.zeros((self.config.height, self.config.width), dtype=np.uint8)
        p2d[:,0] = np.clip(p2d[:,0], 0 ,self.config.height)
        p2d[:,1] = np.clip(p2d[:,1], 0 ,self.config.width)
        mask[p2d[:,0], p2d[:,1]] = 255
        return mask
    
    def get_bbox_masked_cloud(self, mask, depth, rmin=0, rmax=None, cmin=0, cmax=None):
        if not rmax or not cmax:
            rmax = self.config.height
            cmax = self.config.width
        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
        depth_z = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        pt2 = depth_z / self.config.cam_scale
        pt0 = (ymap_masked - self.config.cam_cx) * pt2 / self.config.cam_fx
        pt1 = (xmap_masked - self.config.cam_cy) * pt2 / self.config.cam_fy
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)
        cloud = cloud / 1000.0
        return cloud
    
    def random_downsample(self, cloud, sample_num):
        if len(cloud) > sample_num:
            c_mask = np.zeros(len(cloud), dtype=int)
            c_mask[:sample_num] = 1
            np.random.shuffle(c_mask)
            dcloud = cloud[c_mask.nonzero()]
        else:
            dcloud = np.pad(cloud, (0, sample_num - len(cloud)), 'wrap')
        return dcloud

    def estimate_density(self, cloud, desire_pcd_num):
        origin_size = len(cloud)
        ratio = desire_pcd_num / origin_size
        density_cloud = self.random_downsample(cloud, desire_pcd_num)
        cluster_label = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_pts * ratio).fit_predict(density_cloud)
        cluster_0 = np.where(cluster_label == 0)[0]
        cloud_0 = density_cloud[cluster_0]
        if len(cloud_0) <= 0:
            return self.config.default_voxel_size
        max_x, max_y, max_z = np.max(cloud_0, axis=0)
        min_x, min_y, min_z = np.min(cloud_0, axis=0)
        voxel_size = np.cbrt((max_z - min_z) * (max_y - min_y) * (max_x - min_x) / len(cloud_0)) / 5
        return voxel_size

    def octree_downsample(self, cloud, sample_num):
        voxel_size = self.config.default_voxel_size
        if self.config.estimate_voxel_size:
            voxel_size = self.estimate_density(cloud, sample_num)
        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(cloud)
        res_pcd = source_pcd
        ite = 0
        if self.verbose:
            print(f"input voxel size: {voxel_size}")
        while voxel_size:
            ite += 1
            # octree = o3d.geometry.Octree(max_depth=tree_depth)
            # octree.convert_from_point_cloud(source_pcd)
            sampled_pcd = res_pcd.voxel_down_sample(voxel_size)
            voxel_size *= 1.2
            if (len(sampled_pcd.points) < sample_num):
                if self.verbose:
                    print(f"too few points: {len(sampled_pcd.points)}")
                break
            res_pcd = sampled_pcd
        res_array = np.asarray(res_pcd.points)
        if self.verbose:
            print(f"final voxel size: {voxel_size / 1.2}, iteration {ite}, pcd size: {len(res_pcd.points)}")
            print(f"voxel downsampled result: {len(source_pcd.points)} -> {len(res_pcd.points)}, voxel size: {voxel_size / 1.2}")
            print(f"randomly choose to {sample_num} pc")
        return self.random_downsample(res_array, sample_num)

    def dbscan_tune_iteration(self, cloud):
        range_eps = np.array([0.8, 0.9, 1.0, 1.1, 1.2]) * self.dbscan_eps
        range_points = np.array([0.8, 0.9, 1.0, 1.1, 1.2]) * self.dbscan_min_pts
        max_score, opt_eps, opt_num_points = 0, 0, 0
        for i in range_eps:
            for j in range_points:
                db = DBSCAN(eps = i, min_samples = j).fit(cloud)
                core_sample_mask = np.zeros_like(db.labels_, dtype = bool)
                core_sample_mask[db.core_sample_indices_] = True
                labels = db.labels_
                if self.verbose:
                    print(set(labels))
                silhouette_avg = silhouette_score(cloud, labels)
                if silhouette_avg > max_score:
                    max_score = silhouette_avg
                    opt_eps = i
                    opt_num_points = j
        if self.verbose:
            print('max score is: ', max_score)
            print('best eps is: ', opt_eps)
            print('best num points is: ', opt_num_points)
        self.dbscan_eps = opt_eps
        self.dbscan_min_pts = opt_num_points
        return opt_eps, opt_num_points
    
    def dbscan_pcd_to_label(self, cloud, downsample=True, voxel_downsample=True, tune_scan_param=False):
        """Return mask array with int8 dtype
        """
        origin_size = len(cloud)
        # downsample
        if downsample and len(cloud) > self.config.downsample_num:
            if voxel_downsample:
                cloud = self.octree_downsample(cloud, self.config.downsample_num)
            else:
                cloud = self.random_downsample(cloud, self.config.downsample_num)

        if tune_scan_param:
            self.dbscan_tune_iteration(cloud)

        cluster_label = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_pts * len(cloud) / origin_size).fit_predict(cloud)
        if np.max(cluster_label) < 0:
            cluster_label.fill(0)
            print(f"DBSCAN failed for object {obj} id {rank}")

        most_center_cloud = None
        most_center_distance = 1e9
        num_clusters = np.max(cluster_label) + 1
        for i in range(num_clusters):
            cluster_i = np.where(cluster_label == i)[0]
            cloud_i = cloud[cluster_i]
            cloud_mean = np.mean(cloud_i, axis=0)
            dis_i = np.linalg.norm(cloud_mean[:2])
            if cloud_mean[2] > 20:
                continue
            if dis_i < most_center_distance:
                most_center_distance = dis_i
                most_center_cloud = cloud_i
        
        # # visulization
        # colors = plt.get_cmap("tab20")(cluster_label / (num_clusters - 1 if num_clusters > 1 else 1))
        # colors[cluster_label < 0] = 0
        # print(cloud.shape, colors.shape)
        # fig = go.Figure(
        #     data=[
        #         go.Scatter3d(
        #             x=cloud[:,0], y=cloud[:,1], z=cloud[:,2], 
        #             mode='markers',
        #             marker=dict(size=1, color=colors)
        #         )
        #     ],
        #     layout=dict(
        #         scene=dict(
        #             xaxis=dict(visible=False),
        #             yaxis=dict(visible=False),
        #             zaxis=dict(visible=False)
        #         )
        #     )
        # )
        # fig.show()

        dbscan_label = self.project_3d_mask(most_center_cloud)
        return dbscan_label


    def dbscan_single_frame(self, index, downsample=True, voxel_downsample=False, tune_scan_param=False):
        depth = np.array(Image.open(self.list_depth[index]))
        label = np.array(Image.open(self.list_label[index]))
        np.place(label, label > 0, 255)

        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
        mask_label = ma.getmaskarray(ma.masked_equal(label, np.array(255)))
 
        mask = mask_label * mask_depth

        contour = np.where(mask == True)
        rmin = np.min(contour[0])
        rmax = np.max(contour[0])
        cmin = np.min(contour[1])
        cmax = np.max(contour[1])

        cloud = self.get_bbox_masked_cloud(mask, depth, rmin, rmax, cmin, cmax)

        dbscan_label = self.dbscan_pcd_to_label(cloud, downsample, voxel_downsample, tune_scan_param)

    def reset_param(self):
        self.dbscan_eps = self.reset_eps
        self.dbscan_min_pts = self.reset_min_pts

    def tune_param(self, num_data=1):
        size = min(num_data, len(self.list_depth))
        for i in range(size):
            self.dbscan_single_frame(i, tune_scan_param=True)
        print(f"DBSCAN param after tuning: eps {self.dbscan_eps}, min_sample {self.dbscan_min_pts}")

if __name__ == "__main__":
    label = [f'mseg-semantic/temp_files/mask/9.png']
    depth = [f'dataset/UE-generated/0/depth/9.png']
    config = Configuration("test")
    scanner = Dbscanner(depth, label, config)
    scanner.dbscan_single_frame(0, downsample=True, voxel_downsample=config.voxel_downsample)