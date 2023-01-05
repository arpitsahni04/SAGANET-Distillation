
class Configuration:
    def __init__(self, train=False):
        self.train = train
        self.debug = False
        self.dataset_root = 'dataset/ue_gen_2'
        self.segment_root = 'mseg-semantic/temp_files/mask'
        self.complete_data_root = 'dataset/complete_pcd'
        self.prior_data_root = 'dataset/compare_lib'
        self.object_list = []
        self.object_count = 3
        self.num_per_object = 1000
        if train:
            with open(f"dataset/train_list_sort.txt", 'r') as f:
                    for _ in range(self.object_count):
                        self.object_list.append(int(f.readline().strip()))
        else:
            with open(f"dataset/test_list_sort.txt", 'r') as f:
                    for _ in range(self.object_count):
                        self.object_list.append(int(f.readline().strip()))

        self.pcd_points_num = 2048


        # camera
        self.cam_cx = 320
        self.cam_cy = 240
        self.cam_fx = 320
        self.cam_fy = 320
        self.height = 480
        self.width = 640
        self.cam_scale = 1.0

        # downsample and dbscan
        self.use_dbscan = False
        self.downsample_num = 6000
        self.tune_param = False
        self.tune_number = 10
        self.dbscan_eps = 2
        self.dbscan_min_points = 100
        self.voxel_downsample = True
        self.estimate_voxel_size = True
        self.default_voxel_size = 0.03

        # pose eval
        # models
        self.pose_model = 'trained_ckps/pose_model_109_0.6639944425473611.pth'
        self.pose_refine_model = 'trained_ckps/pose_refine_model_492_0.03314094381348696.pth'
        self.refine_iters = 0 # set 0 if not use refinement model
        # If use FPFH, will not use network to estimate the pose.
        # Do NOT support FPFH in the pipeline
        self.use_fpfh = False 
        self.use_icp = True

        # completion
        self.process_for_prior = True
        self.partial_pcd_num = 2048
        self.complete_pcd_num = 1024
        self.prior_num = 2048
        self.crop_point_num = 1024
        # models
        self.netG = 'Trained_Model/gen_net_Table_Attention1.pth'
        self.netD = 'Trained_Model/dis_net_Table_Attention1.pth'
        self.visualize = True

        # output path
        self.output_model_dir = 'trained_models'
        self.output_train_log_dir = 'experiments/logs'
        self.output_result_dir = 'experiments/logs'
