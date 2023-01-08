import open3d as o3d
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import json
import codecs
import numpy as np
import sys
import torchvision.transforms as transforms
import argparse
import json
import random
import numpy.ma as ma
import copy
import scipy.misc
import scipy.io as scio
from scipy.spatial.distance import directed_hausdorff
from sklearn.cluster import DBSCAN
import yaml
import cv2
from dataset.dbscan import Dbscanner
from config import Configuration

def read_pcd(filename):
    pcd = o3d.io.read_point_cloud(filename, format = 'pcd')
    return np.array(pcd.points) #* 10

def get_prior_set(complete_dir):
    complete_pcds = []
    complete_dir = os.path.abspath(complete_dir)
    for f in os.listdir(complete_dir):
        complete_pcd = read_pcd(os.path.join(f"{complete_dir}/{f}"))
        complete_pcd = resample_pcd(complete_pcd, 2048)
        complete_pcds.append(complete_pcd)
    #partial_pcds = [torch.from_numpy(partial_pcd1), torch.from_numpy(partial_pcd2), torch.from_numpy(partial_pcd3)]
    return complete_pcds

def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size = n - pcd.shape[0])])
    return pcd[idx[:n]]

def get_prior(p_cloud, c_clouds):
    #criterion_PointLoss = PointLoss()
    #p_cloud is partial

    cd_loss = 1000
    partial_pcd = p_cloud
    fin_pcd = p_cloud
    for idx in range(len(c_clouds)):
        # c_clouds[idx] += 5
        cd_loss_tmp = directed_hausdorff(partial_pcd, c_clouds[idx])[0]
        if cd_loss_tmp < cd_loss:
            cd_loss = cd_loss_tmp
            fin_pcd = c_clouds[idx]
            fin_idx = idx
    
    return fin_pcd.astype(np.float32), fin_idx, cd_loss

# https://medium.com/@kidargueta/normalizing-feature-scaling-point-clouds-for-machine-learning-8138c6e69f5

def normalize_pc(points):
	centroid = np.mean(points, axis=0)
	points -= centroid
	furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2,axis=-1)))
	points /= furthest_distance

	return points


class FullDataset(data.Dataset):
    def __init__(self, mode, refine=True):
        self.mode = mode
        self.config = Configuration(mode == 'train')

        self.list_rgb = []
        self.list_depth = []
        self.list_label = []
        self.list_obj = []
        self.list_rank = []
        self.poses = []
        self.complete_pcd = []
        self.list_pcd = []
        self.list_complete_pcd = []
        self.refine = refine

        self.obj_list = self.config.object_list
        
        if self.mode == 'train':
            image_idx = []
            for i in range(self.config.num_per_object // 5):
                #800 images train set per class
                image_idx.append(5 * i)
                image_idx.append(5 * i + 1)
                image_idx.append(5 * i + 2)
                image_idx.append(5 * i + 3)
        else:
            #test set has every 5th image 200 images test set per class
            image_idx = [i * 5 + 4 for i in range(self.config.num_per_object // 5)]

        for item in self.obj_list:
            for idx in image_idx:
                self.list_rgb.append(f'{self.config.dataset_root}/{item}/rgb/{idx}.png')
                self.list_depth.append(f'{self.config.dataset_root}/{item}/depth/{idx}.png')

                self.list_label.append(f'{self.config.dataset_root}/{item}/mask/{idx}.png')
                # self.list_label.append(f'{self.config.segment_root}/{idx}.png')
                self.list_obj.append(item)
                self.list_rank.append(idx)

                self.poses.append(np.loadtxt(f'{self.config.dataset_root}/{item}/pose/{idx}.txt'))

            self.complete_pcd.append(np.asarray(o3d.io.read_point_cloud(f'{self.config.complete_data_root}/{item}.pcd').points))
            
            print("Object {0} buffer loaded".format(item))

        self.length = len(self.list_rgb)
        # to get indices i,j
        self.xmap = np.array([[j for _ in range(self.config.width)] for j in range(self.config.height)])
        self.ymap = np.array([[i for i in range(self.config.width)] for _ in range(self.config.height)])
        
        self.num = self.config.pcd_points_num
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.num_pt_mesh_large = 2048
        self.num_pt_mesh_small = 2048 # whhy equal?
        self.symmetry_obj_idx = []

        self.dbscanner = Dbscanner(self.list_depth, self.list_label, self.config)
        self.prior_set = get_prior_set(self.config.prior_data_root)

    def __getitem__(self, index):
        img = Image.open(self.list_rgb[index])
        # ori_img = np.array(img) # not used in pc
        depth = np.array(Image.open(self.list_depth[index])) # 1 channels
        label = np.array(Image.open(self.list_label[index])) # 1 channel
        np.place(label, label > 0, 255)
        obj = self.list_obj[index]
        #rank = self.list_rank[index] # rank represents idx in the init function above, not used in code   

        pose = self.poses[index]

        img = np.array(img)[:, :, :3] # PIL to numpy array
        img = np.transpose(img, (2, 0, 1)) # channel, height,width
        img_masked = img 

        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0)) # true where not equal 0 
        # getmaskarray converts 0,1 to booleans
        mask_label = ma.getmaskarray(ma.masked_equal(label, np.array(255)))
        mask = mask_label * mask_depth
        # returns true when both true else flase
        contour = np.where(mask == True)
        rmin = np.min(contour[0]) # top most row
        rmax = np.max(contour[0]) # bottom most row
        cmin = np.min(contour[1])
        cmax = np.max(contour[1])
        # find effective bounding box
        if self.config.use_dbscan:
            choose, cloud = self.get_bbox_masked_cloud(mask, depth, rmin, rmax, cmin, cmax, None)
            dbscan_label = self.dbscanner.dbscan_pcd_to_label(cloud, downsample=True, voxel_downsample=self.config.voxel_downsample)
            mask_label = ma.getmaskarray(ma.masked_equal(dbscan_label, np.array(255)))
            mask = mask_label * mask_depth
            # cv2.imwrite(f"experiments/visualization/dbscan/{obj}_{rank}.png", dbscan_label)

            contour = np.where(mask == True)
            rmin = np.min(contour[0])
            rmax = np.max(contour[0])
            cmin = np.min(contour[1])
            cmax = np.max(contour[1])

        img_masked = img_masked[:, rmin:rmax, cmin:cmax]
        # p_img = np.transpose(img_masked, (1, 2, 0))
        # cv2.imwrite('{0}_input.png'.format(index), p_img) #scipy.misc.imsave

        target_r = np.array(pose[:3,:3])
        target_t = np.array(pose[:3,3]) / 10 # to go to blender coordinates

        choose, cloud = self.get_bbox_masked_cloud(mask, depth, rmin, rmax, cmin, cmax, self.num)
        cloud = cloud / 10 # so complete is blender, unreal points need to be scaled down

        # use complete pcd and downsample it
        model_points = self.complete_pcd[obj]
        # dellist_idx = [j for j in range(0, len(model_points))]
        # dellist_idx = random.sample(dellist_idx, len(model_points) - self.num_pt_mesh_small)
        # model_points = np.delete(model_points, dellist_idx, axis=0)
        # print("model_points: ", model_points.shape)
        # Below is based on Blender generated data, model size is about 0.7 * 0.4 * 0.4 meter
        # target = np.add(model_points, -target_t)
        # target = np.dot(target, np.dot(target_r, np.array([[1, 0, 0],[0, -1, 0],[0, 0, -1]])))

        # Below is based on the ground truth we got from Airsim, which is based on NED coordinates
        # model size is about 7 * 4 * 4 meter

        # partial to complete pose
        cam_ned_wrt_obj_ned = np.concatenate((target_r, np.array([target_t]).T), axis=1)
        cam_ned_wrt_obj_ned = np.concatenate((cam_ned_wrt_obj_ned, np.array([[0, 0, 0, 1]])), axis=0)
        #?
        obj_ned_wrt_obj = np.array([[1, 0, 0, 0],
                                    [0, 0, -1, 0],
                                    [0, 1, 0, 0],
                                    [ 0, 0, 0, 1]])
        cam_real_wrt_cam_ned = np.array([[0, 0, 1, 0],
                                         [1, 0, 0, 0],
                                         [0, 1, 0, 0],
                                         [0, 0, 0, 1]]) 
        # add some noise later
        cam_wrt_obj = obj_ned_wrt_obj @ cam_ned_wrt_obj_ned @ cam_real_wrt_cam_ned
        obj_wrt_cam = np.linalg.inv(cam_wrt_obj)
        # The correct output prediction should be obj_wrt_cam
        truth_r = obj_wrt_cam[:3,:3]
        truth_t = obj_wrt_cam[:3,3]
        target = np.dot(model_points, truth_r.T) + truth_t # would this be equal to P = T*[X Y Z 1]^T ?
        
        if self.config.process_for_prior:
            cloud = np.dot(cloud, cam_wrt_obj[:3, :3].T) + cam_wrt_obj[:3, 3] 

            cloud = resample_pcd(cloud, self.config.partial_pcd_num)
            cloud_pcd = o3d.geometry.PointCloud()
            cloud_pcd.points = o3d.utility.Vector3dVector(cloud)
            o3d.io.write_point_cloud("/home/arpitsah/Desktop/Projects Fall-22/ShapeAware-Pipeline/completion/dataset/modelpoints_new/cloudpoints_new/cloud_points{}.ply".format(index), cloud_pcd)

            model_points = resample_pcd(model_points, self.config.complete_pcd_num) # add to 189
            model_points_pcd = o3d.geometry.PointCloud()
            model_points_pcd.points = o3d.utility.Vector3dVector(model_points)
            o3d.io.write_point_cloud("/home/arpitsah/Desktop/Projects Fall-22/ShapeAware-Pipeline/completion/dataset/modelpoints_new/model_points_{}.ply".format(index), model_points_pcd)
            
            return cloud, model_points

        return torch.from_numpy(cloud.astype(np.float32)), \
               torch.LongTensor(choose.astype(np.int32)), \
               self.norm(torch.from_numpy(img_masked.astype(np.float32))), \
               torch.from_numpy(target.astype(np.float32)), \
               torch.from_numpy(model_points.astype(np.float32)), \
               torch.LongTensor([self.obj_list.index(obj)])

    def __len__(self):
        return self.length

    def get_sym_list(self):
        return self.symmetry_obj_idx

    def get_num_points_mesh(self):
        if self.refine:
            return self.num_pt_mesh_large
        else:
            return self.num_pt_mesh_small

    def get_bbox_masked_cloud(self, mask, depth, rmin=0, rmax=None, cmin=0, cmax=None, num_points=None):
        if not rmax or not cmax:
            # basically if vehicle not in image or countour bounds not found
            rmax = self.config.height
            cmax = self.config.width
        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0] # nonzero outputs a tuple 
        # if num_points is given, choose the exact number of point clouds
        if num_points and len(choose) > num_points:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:num_points] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        elif num_points:
            print("points after downsample and clustering are too few")
            choose = np.pad(choose, (0, num_points - len(choose)), 'wrap')
        depth_z = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        # it moves the origin of the coordinate system to the center of the data
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        pt2 = depth_z / self.config.cam_scale
        pt0 = (ymap_masked - self.config.cam_cx) * pt2 / self.config.cam_fx
        pt1 = (xmap_masked - self.config.cam_cy) * pt2 / self.config.cam_fy
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)
        # y,x,z ?
        cloud = cloud / 1000.0
        return choose, cloud
