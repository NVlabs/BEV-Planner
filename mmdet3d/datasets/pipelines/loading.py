# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion
import os.path as osp
from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
from ...core.bbox import LiDARInstance3DBoxes
from ..builder import PIPELINES
from copy import deepcopy
import cv2
import os
from torchvision.transforms.functional import rotate
from mmdet3d.datasets.vector_map import VectorizedLocalMap, LiDARInstanceLines
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.eval.common.utils import Quaternion as Quaternion_nus
# from .vad_custom_nuscenes_eval import NuScenesEval_custom
from nuscenes.eval.common.utils import center_distance
from mmcv.parallel import DataContainer as DC
import random
from mmdet3d.core import LiDARInstance3DBoxes
from nuscenes.utils.data_classes import Box as NuScenesBox
# from projects.mmdet3d_plugin.core.bbox.structures.nuscenes_box import CustomNuscenesBox
from shapely import affinity, ops
from shapely.geometry import LineString, box, MultiPolygon, MultiLineString
from mmdet.datasets.pipelines import to_tensor
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.eval.detection.constants import DETECTION_NAMES
from mmcv.runner import get_dist_info
from nuscenes.utils.data_classes import Box as NuScenesBox
import pyquaternion
import torch.nn as nn

@PIPELINES.register_module()
class LoadVectorMap(object):

    def __init__(self, data_root, point_cloud_range, map_fixed_ptsnum_per_line=20, map_classes=['divider', 'ped_crossing', 'boundary'], **kwargs):
        patch_h = point_cloud_range[4]-point_cloud_range[1]
        patch_w = point_cloud_range[3]-point_cloud_range[0]
        self.patch_size = (min(patch_h, 50), patch_w)
        self.vector_map = VectorizedLocalMap(data_root,  patch_size=self.patch_size, map_classes=map_classes, 
                            fixed_ptsnum_per_line=map_fixed_ptsnum_per_line)


    def vectormap_pipeline(self, location, ego2global_translation, patch_angle, flip_dx, flip_dy):
        '''
        `example` type: <class 'dict'>
            keys: 'img_metas', 'gt_bboxes_3d', 'gt_labels_3d', 'img';
                  all keys type is 'DataContainer';
                  'img_metas' cpu_only=True, type is dict, others are false;
                  'gt_labels_3d' shape torch.size([num_samples]), stack=False,
                                padding_value=0, cpu_only=False
                  'gt_bboxes_3d': stack=False, cpu_only=True
        '''

        anns_results = self.vector_map.gen_vectorized_samples(
            location, ego2global_translation, patch_angle, flip_dx, flip_dy
        )
        has_valid_map = True
        if len(anns_results['gt_vecs_label']) == 0:    
            ## params that can generate non-empty anns
            location = 'boston-seaport'
            ego2global_translation = [1178.1282, 1140.1135, 0.0]
            patch_angle = 143.6049566307475
            flip_dx = False
            flip_dy = False
            ## 
            pseudo_anns_results = self.vector_map.gen_vectorized_samples(
                location, ego2global_translation, patch_angle, flip_dx, flip_dy
            )
            anns_results = pseudo_anns_results
            has_valid_map = False

        

        '''
        anns_results, type: dict
            'gt_vecs_pts_loc': list[num_vecs], vec with num_points*2 coordinates
            'gt_vecs_pts_num': list[num_vecs], vec with num_points
            'gt_vecs_label': list[num_vecs], vec with cls index
        '''
        gt_vecs_label = to_tensor(anns_results['gt_vecs_label'])
        if isinstance(anns_results['gt_vecs_pts_loc'], LiDARInstanceLines):
            gt_vecs_pts_loc = anns_results['gt_vecs_pts_loc']
        else:
            gt_vecs_pts_loc = to_tensor(anns_results['gt_vecs_pts_loc'])
            try:
                gt_vecs_pts_loc = gt_vecs_pts_loc.flatten(1).to(dtype=torch.float32)
            except:
                assert False
                # empty tensor, will be passed in train, 
                # but we preserve it for test
                gt_vecs_pts_loc = gt_vecs_pts_loc

        return dict(
            map_gt_labels_3d = DC(gt_vecs_label, cpu_only=False),
            map_gt_bboxes_3d = DC(gt_vecs_pts_loc, cpu_only=True),
            has_valid_map = has_valid_map,
        )

    def __call__(self, results):

        ego2global_translation = list(results['ori_ego_pose'][:3,3].numpy())
        # ego2global_rotation = list(Quaternion_nus(matrix=ego2global.numpy(), rtol=eps, atol=eps).q)
        v = np.dot( results['ori_ego_pose'][:3,:3].numpy(), np.array([1, 0, 0]))
        yaw = np.arctan2(v[1], v[0])
        ori_patch_angle = yaw / np.pi * 180

        # v = np.dot(ego2global[:3,:3].numpy(), np.array([1, 0, 0]))
        # yaw = np.arctan2(v[1], v[0])
        # patch_angle2 = yaw / np.pi * 180

        results.update(
            self.vectormap_pipeline(results['curr']['map_location'], ego2global_translation, ori_patch_angle-results['rotate_bda'], results['flip_dx'], results['flip_dy'])
        )
        return results


@PIPELINES.register_module()
class LoadVectorMap2(object):

    def __init__(self, data_root, point_cloud_range, map_fixed_ptsnum_per_line=20, map_classes=['divider', 'ped_crossing', 'boundary'], **kwargs):
        patch_h = point_cloud_range[4]-point_cloud_range[1]
        patch_w = point_cloud_range[3]-point_cloud_range[0]
        self.point_cloud_range = torch.tensor(point_cloud_range)
        self.patch_size = (min(patch_h, 50), patch_w)
        self.vector_map = VectorizedLocalMap(data_root,  patch_size=self.patch_size, map_classes=map_classes, 
                            fixed_ptsnum_per_line=map_fixed_ptsnum_per_line)


    def vectormap_pipeline(self, location, ego2global_translation, patch_angle, flip_dx, flip_dy):
        '''
        `example` type: <class 'dict'>
            keys: 'img_metas', 'gt_bboxes_3d', 'gt_labels_3d', 'img';
                  all keys type is 'DataContainer';
                  'img_metas' cpu_only=True, type is dict, others are false;
                  'gt_labels_3d' shape torch.size([num_samples]), stack=False,
                                padding_value=0, cpu_only=False
                  'gt_bboxes_3d': stack=False, cpu_only=True
        '''

        anns_results = self.vector_map.gen_vectorized_samples(
            location, ego2global_translation, patch_angle, flip_dx, flip_dy
        )
        has_valid_map = True
        if len(anns_results['gt_vecs_label']) == 0:    
            ## params that can generate non-empty anns
            location = 'boston-seaport'
            ego2global_translation = [1178.1282, 1140.1135, 0.0]
            patch_angle = 143.6049566307475
            flip_dx = False
            flip_dy = False
            ## 
            pseudo_anns_results = self.vector_map.gen_vectorized_samples(
                location, ego2global_translation, patch_angle, flip_dx, flip_dy
            )
            anns_results = pseudo_anns_results
            has_valid_map = False

        

        '''
        anns_results, type: dict
            'gt_vecs_pts_loc': list[num_vecs], vec with num_points*2 coordinates
            'gt_vecs_pts_num': list[num_vecs], vec with num_points
            'gt_vecs_label': list[num_vecs], vec with cls index
        '''
        gt_vecs_label = to_tensor(anns_results['gt_vecs_label'])
        if isinstance(anns_results['gt_vecs_pts_loc'], LiDARInstanceLines):
            gt_vecs_pts_loc = anns_results['gt_vecs_pts_loc']
        else:
            gt_vecs_pts_loc = to_tensor(anns_results['gt_vecs_pts_loc'])
            try:
                gt_vecs_pts_loc = gt_vecs_pts_loc.flatten(1).to(dtype=torch.float32)
            except:
                assert False
                # empty tensor, will be passed in train, 
                # but we preserve it for test
                gt_vecs_pts_loc = gt_vecs_pts_loc

        gt_pts = gt_vecs_pts_loc.shift_fixed_num_sampled_points_v2
        gt_pts = (gt_pts - self.point_cloud_range[:2])/(self.point_cloud_range[3:5]-self.point_cloud_range[:2])
        return dict(
            map_gt_labels_3d = DC(gt_vecs_label, cpu_only=False),
            map_gt_bboxes_3d = DC(gt_pts, cpu_only=False),
            has_valid_map = has_valid_map,
        )

    def __call__(self, results):

        ego2global_translation = list(results['ori_ego_pose'][:3,3].numpy())
        # ego2global_rotation = list(Quaternion_nus(matrix=ego2global.numpy(), rtol=eps, atol=eps).q)
        v = np.dot( results['ori_ego_pose'][:3,:3].numpy(), np.array([1, 0, 0]))
        yaw = np.arctan2(v[1], v[0])
        ori_patch_angle = yaw / np.pi * 180

        # v = np.dot(ego2global[:3,:3].numpy(), np.array([1, 0, 0]))
        # yaw = np.arctan2(v[1], v[0])
        # patch_angle2 = yaw / np.pi * 180

        results.update(
            self.vectormap_pipeline(results['curr']['map_location'], ego2global_translation, ori_patch_angle-results['rotate_bda'], results['flip_dx'], results['flip_dy'])
        )
        return results


@PIPELINES.register_module()
class LoadGTPlaner(object):
    def __init__(self):
        pass
    
    def __call__(self, results):

        results['gt_ego_lcf_feat'] = to_tensor(results['curr']['gt_ego_lcf_feat'])
        results['gt_ego_lcf_feat'][:2] = (results['bda_mat'][:2, :2] @ results['gt_ego_lcf_feat'][:2, None]).squeeze(-1)
        results['gt_ego_fut_trajs'] = torch.cumsum(to_tensor(results['curr']['gt_ego_fut_trajs']), dim=0)[:6]
        results['gt_ego_fut_trajs'] = (results['bda_mat'][:2,:2] @ results['gt_ego_fut_trajs'][..., None]).squeeze(-1)
        results['gt_ego_his_trajs'] = -to_tensor(results['curr']['gt_ego_his_trajs'])
        results['gt_ego_his_trajs'] = (results['bda_mat'][:2,:2] @ results['gt_ego_his_trajs'][..., None]).squeeze(-1)
        if results['gt_ego_fut_trajs'][-1][1] >= 2:
            command = np.array([1, 0, 0])  # Turn Right
        elif results['gt_ego_fut_trajs'][-1][1] <= -2:
            command = np.array([0, 1, 0])  # Turn Left
        else:
            command = np.array([0, 0, 1])  # Go Straight
        
        results['gt_ego_fut_cmd'] = to_tensor(command)
        results['gt_ego_fut_masks'] = to_tensor(results['curr']['gt_ego_fut_masks'])[: 6]
        return results



@PIPELINES.register_module()
class LoadGTMotion(object):
    def __init__(self, with_ego_as_agent=False):
        self.with_ego_as_agent = with_ego_as_agent
    
    def __call__(self, results):

        agent_fut_traj_mask = torch.tensor(np.array(results['curr']['ann_infos']['fut_traj_mask']), dtype=torch.float32)
        agent_fut_traj = torch.tensor(np.array(results['curr']['ann_infos']['fut_traj']), dtype=torch.float32)
        agent_fut_traj = torch.cat([agent_fut_traj, torch.ones_like(agent_fut_traj[..., 0:2])], dim=-1)
        if len(agent_fut_traj)>0:
            agent_fut_traj = (results['ego_pose_inv'] @ agent_fut_traj.unsqueeze(-1)).squeeze(-1)[..., :2] * agent_fut_traj_mask
            
        if self.with_ego_as_agent:
            gt_ego_fut_trajs = torch.cumsum(to_tensor(results['curr']['gt_ego_fut_trajs']), dim=0)[: 6]
            gt_ego_fut_trajs = torch.cat([gt_ego_fut_trajs, torch.zeros_like(gt_ego_fut_trajs[:2])])
            agent_fut_traj = torch.cat([gt_ego_fut_trajs[None], agent_fut_traj], 0)
            
            gt_fut_traj_mask = torch.ones_like(gt_ego_fut_trajs)
            gt_fut_traj_mask[-2:] = 0
            agent_fut_traj_mask = torch.cat([gt_fut_traj_mask[None], agent_fut_traj_mask], 0)

        centers = results['gt_bboxes_3d'].center[..., :2]
        try:
            tmp = torch.cat([centers[:, None], agent_fut_traj], 1)
        except:
            print(centers.shape, agent_fut_traj.shape, agent_fut_traj_mask.shape, results['gt_labels_3d'].shape)
        agent_fut_traj = tmp[:, 1:] - tmp[:, :-1]
        results['gt_agent_fut_traj_mask'] = agent_fut_traj_mask
        results['gt_agent_fut_traj'] = agent_fut_traj

        return results


@PIPELINES.register_module()
class LoadFutBoxInfo(object):
    def __init__(self, add_boundary=True):
        self.X_BOUND = [-50.0, 50.0, 0.1]  # Forward
        self.Y_BOUND = [-50.0, 50.0, 0.1]  # Sides
        self.Z_BOUND = [-10.0, 10.0, 20.0]  # Height
        dx, bx, _ = self.gen_dx_bx(self.X_BOUND, self.Y_BOUND, self.Z_BOUND)
        self.dx, self.bx = dx[:2], bx[:2]
    
        bev_resolution, bev_start_position, bev_dimension = self.calculate_birds_eye_view_parameters(
            self.X_BOUND, self.Y_BOUND, self.Z_BOUND
        )
        self.bev_resolution = bev_resolution.numpy()
        self.bev_start_position = bev_start_position.numpy()
        self.bev_dimension = bev_dimension.numpy()
        ego_width, ego_length = 1.85, 4.084
        self.W = ego_width
        self.H = ego_length

        self.category_index = {
            'human':[2,3,4,5,6,7,8],
            'vehicle':[14,15,16,17,18,19,20,21,22,23]
        }
        self.add_boundary = add_boundary
        # self.n_future = n_future

        # self.add_state("obj_col", default=torch.zeros(self.n_future), dist_reduce_fx="sum")
        # self.add_state("obj_box_col", default=torch.zeros(self.n_future), dist_reduce_fx="sum")
        # self.add_state("L2", default=torch.zeros(self.n_future),dist_reduce_fx="sum")
        # self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def gen_dx_bx(self, xbound, ybound, zbound):
        dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
        bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
        nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])

        return dx, bx, nx
    
    def calculate_birds_eye_view_parameters(self, x_bounds, y_bounds, z_bounds):
        """
        Parameters
        ----------
            x_bounds: Forward direction in the ego-car.
            y_bounds: Sides
            z_bounds: Height

        Returns
        -------
            bev_resolution: Bird's-eye view bev_resolution
            bev_start_position Bird's-eye view first element
            bev_dimension Bird's-eye view tensor spatial dimension
        """
        bev_resolution = torch.tensor([row[2] for row in [x_bounds, y_bounds, z_bounds]])
        bev_start_position = torch.tensor([row[0] + row[2] / 2.0 for row in [x_bounds, y_bounds, z_bounds]])
        bev_dimension = torch.tensor([(row[1] - row[0]) / row[2] for row in [x_bounds, y_bounds, z_bounds]],
                                    dtype=torch.long)

        return bev_resolution, bev_start_position, bev_dimension
    
    def get_label(
            self,
            boxes_in_cur_ego_list,
            labels_in_cur_ego_list
        ):
        segmentation_np, pedestrian_np = self.get_birds_eye_view_label(boxes_in_cur_ego_list, labels_in_cur_ego_list)
        segmentation = torch.from_numpy(segmentation_np).long()
        pedestrian = torch.from_numpy(pedestrian_np).long()


        return segmentation, pedestrian

    def world2bev_vis(self, x, y):
            return int((x - self.bx[0].item()) / self.dx[0].item()), int((y - self.bx[1].item()) / self.dx[1].item())
    
    def get_birds_eye_view_label(self, boxes_in_cur_ego_list, labels_in_cur_ego_list):
        T = 6
        segmentation = np.zeros((T,self.bev_dimension[0], self.bev_dimension[1]))
        pedestrian = np.zeros((T,self.bev_dimension[0], self.bev_dimension[1]))

        for k, fut_boxes in enumerate(boxes_in_cur_ego_list):
            if fut_boxes is None: continue
            for i, corners in enumerate(fut_boxes.corners[:, [4, 7, 3, 0], :2]):
                
                # fitler vehicle
                vehicle_classes = ['car', 'bus', 'construction_vehicle',
                           'bicycle', 'motorcycle', 'truck', 'trailer']
                if labels_in_cur_ego_list[k][i] not in  [0, 1, 2, 3, 4, 6, 7]: continue 
                corners = np.array([self.world2bev_vis(*corner) for corner in corners])
                cv2.fillPoly(segmentation[k], [corners], 1.0)
                
        return segmentation, pedestrian

    def __call__(self, results):

        ego2global_rotation = results['nuscenes_get_rt_matrix']['ego2global_rotation']
        ego2global_translation =results['nuscenes_get_rt_matrix'][
            'ego2global_translation']
        trans = -np.array(ego2global_translation)
        rot = Quaternion(ego2global_rotation).inverse

        boxes_in_cur_ego_list = []
        for gt_boxes_each_frame in results['fut_boxes_info']:
            boxes_in_cur_ego = []
            if len(gt_boxes_each_frame)==0:
                boxes_in_cur_ego_list.append(None)
                continue
            for box in gt_boxes_each_frame:
                center = box[:3]
                wlh = box[3:6]
                box_yaw = box[6]
                box_vel = box[7:].tolist()
                box_vel.append(0)
                quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw)
                nusc_box = NuScenesBox(center, wlh, quat, velocity=box_vel)
                nusc_box.translate(trans)
                nusc_box.rotate(rot)
                box_xyz = np.array(nusc_box.center)
                box_dxdydz = np.array(nusc_box.wlh)
                box_yaw = np.array([nusc_box.orientation.yaw_pitch_roll[0]])
                box_velo = np.array(nusc_box.velocity[:2])
                gt_box = np.concatenate([box_xyz, box_dxdydz, box_yaw, box_velo])
                boxes_in_cur_ego.append(gt_box)
            boxes_in_cur_ego = torch.tensor(np.array(boxes_in_cur_ego))
            boxes_in_cur_ego = LiDARInstance3DBoxes(boxes_in_cur_ego, box_dim=boxes_in_cur_ego.shape[-1],
                                 origin=(0.5, 0.5, 0.5))

            boxes_in_cur_ego_list.append(boxes_in_cur_ego)
            
        results['fut_boxes_in_cur_ego_list'] = boxes_in_cur_ego_list
        segmentation, pedestrian = self.get_label(boxes_in_cur_ego_list, results['fut_labels_info'])

        
        segmentation_plus = segmentation.permute(1, 2, 0).cpu().clone().numpy()
        segmentation_plus *= 0 # only consider boudnary, temporal
        map_gt_bboxes_3d = results['map_gt_bboxes_3d'].data.fixed_num_sampled_points
        map_gt_bboxes_3d= map_gt_bboxes_3d[ results['map_gt_labels_3d'].data==2]
        map_gt_bboxes_3d = (map_gt_bboxes_3d - self.bx.cpu().numpy() ) / (self.dx.cpu().numpy())
        a = segmentation_plus[:, :, :3].copy()
        a = np.ascontiguousarray(a, dtype=np.uint8)
        b = segmentation_plus[:, :, :3].copy()
        b = np.ascontiguousarray(a, dtype=np.uint8)
        for line in map_gt_bboxes_3d:
            line = line.clip(0, 999).numpy().astype(np.int32)
            for i, corner in enumerate(line[:-1]):
                a = cv2.line(a, tuple(line[i]), tuple(line[i+1]), color=(1, 1, 1), thickness=1)
                b = cv2.line(b, tuple(line[i]), tuple(line[i+1]), color=(1, 1, 1), thickness=1)   
        segmentation_plus = torch.cat([torch.tensor(a), torch.tensor(b)], -1).permute(2, 0, 1)

        results['gt_fut_segmentations'] = segmentation
        results['gt_fut_segmentations_plus'] = segmentation_plus
        return results


@PIPELINES.register_module()
class LoadSemanticImageMask(object):
    def __init__(self, mask_file_path='./data/nus_sem'):
        self.mask_file_path = mask_file_path
    
    def __call__(self, results):

        masks = []
        for cam in results['cam_names']:
            data_token = results['curr']['cams'][cam]['sample_data_token']
            filename = osp.join(self.mask_file_path, data_token+'.png')
            img = Image.open(filename)
            img_augs = results['img_augs'][cam]
            resize, resize_dims, crop, flip, rotate = img_augs        
            img = self.img_transform_core(img, resize_dims, crop, flip, rotate)
            img = np.array(img)
            masks.append(img)
        masks = np.stack(masks, 0)
        results['gt_img_sem_masks'] = to_tensor(masks)
        return results
        
    
    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims, resample=0)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate, resample=0, expand=0)
        return img


@PIPELINES.register_module()
class LoadMultiViewImageFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool, optional): Whether to convert the img to float32.
            Defaults to False.
        color_type (str, optional): Color type of the file.
            Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results['img_filename']
        # img is of shape (h, w, c, num_views)
        img = np.stack(
            [mmcv.imread(name, self.color_type) for name in filename], axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formatting.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@PIPELINES.register_module()
class LoadImageFromFileMono3D(object):
    """Load an image from file in monocular 3D object detection. Compared to 2D
    detection, additional camera parameters need to be loaded.

    Args:
        kwargs (dict): Arguments are the same as those in
            :class:`LoadImageFromFile`.
    """


    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        super().__call__(results)
        results['cam2img'] = results['img_info']['cam_intrinsic']
        return results

@PIPELINES.register_module()
class LoadOccupancy(object):
    """Load an image from file in monocular 3D object detection. Compared to 2D
    detection, additional camera parameters need to be loaded.

    Args:
        kwargs (dict): Arguments are the same as those in
            :class:`LoadImageFromFile`.
    """

    def __init__(self, occupancy_path='/mount/dnn_data/occupancy_2023/gts',
                    num_classes=17,
                    ignore_nonvisible=False,
                    mask='mask_camera',
                    ignore_classes=[],
                    fix_void=True) :
        self.occupancy_path = occupancy_path
        self.num_classes = num_classes
        self.ignore_nonvisible = ignore_nonvisible
        self.mask = mask

        self.ignore_classes=ignore_classes

        self.fix_void = fix_void


    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        scene_name = results['curr']['scene_name']
        sample_token = results['curr']['token']


        occupancy_file_path = osp.join(self.occupancy_path, scene_name, sample_token, 'labels.npz')
        data = np.load(occupancy_file_path)
        occupancy = torch.tensor(data['semantics'])
        visible_mask = torch.tensor(data[self.mask])
        # visible_mask_lidar = data['mask_lidar']

        if self.ignore_nonvisible:
            occupancy[~visible_mask.to(torch.bool)] = 255


        # to BEVDet format
        occupancy = occupancy.permute(2, 0, 1)
        occupancy = torch.rot90(occupancy, 1, [1, 2])
        occupancy = torch.flip(occupancy, [1])
        occupancy = occupancy.permute(1, 2, 0)


        if self.fix_void:
            occupancy[occupancy<255] = occupancy[occupancy<255] + 1

        for class_ in self.ignore_classes:
            occupancy[occupancy==class_] = 255

        if results['rotate_bda'] != 0:
            occupancy = occupancy.permute(2, 0, 1)
            occupancy = rotate(occupancy, -results['rotate_bda'], fill=255).permute(1, 2, 0)

        if results['flip_dx']:
            occupancy = torch.flip(occupancy, [1])

        if results['flip_dy']:
            occupancy = torch.flip(occupancy, [0])



        results['gt_occupancy'] = occupancy
        results['visible_mask'] = visible_mask
        
        results['visible_mask_bev'] = (occupancy==255).sum(-1)

        return results


@PIPELINES.register_module()
class LoadPointsFromMultiSweeps(object):
    """Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int, optional): Number of sweeps. Defaults to 10.
        load_dim (int, optional): Dimension number of the loaded points.
            Defaults to 5.
        use_dim (list[int], optional): Which dimension to use.
            Defaults to [0, 1, 2, 4].
        time_dim (int, optional): Which dimension to represent the timestamps
            of each points. Defaults to 4.
        file_client_args (dict, optional): Config dict of file clients,
            refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
        pad_empty_sweeps (bool, optional): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool, optional): Whether to remove close points.
            Defaults to False.
        test_mode (bool, optional): If `test_mode=True`, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(self,
                 sweeps_num=10,
                 load_dim=5,
                 use_dim=[0, 1, 2, 4],
                 time_dim=4,
                 file_client_args=dict(backend='disk'),
                 pad_empty_sweeps=False,
                 remove_close=False,
                 translate2ego=False,
                 test_mode=False):
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        self.use_dim = use_dim
        self.time_dim = time_dim
        assert time_dim < load_dim, \
            f'Expect the timestamp dimension < {load_dim}, got {time_dim}'
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        self.translate2ego = translate2ego
        
    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)
        return points

    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray | :obj:`BasePoints`): Sweep points.
            radius (float, optional): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data.
                Added key and value are described below.

                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point
                    cloud arrays.
        """
        points = results['points']
        points.tensor[:, self.time_dim] = 0
        sweep_points_list = [points]
        ts = results['timestamp']

        if self.pad_empty_sweeps and len(results['sweeps']) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    sweep_points_list.append(self._remove_close(points))
                else:
                    sweep_points_list.append(points)
        else:
            if len(results['sweeps']) <= self.sweeps_num:
                choices = np.arange(len(results['sweeps']))
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                choices = np.random.choice(
                    len(results['sweeps']), self.sweeps_num, replace=False)
            for idx in choices:
                sweep = results['sweeps'][idx]
                points_sweep = self._load_points(sweep['data_path'])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)
                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                sweep_ts = sweep['timestamp'] / 1e6
                points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                    'sensor2lidar_rotation'].T
                points_sweep[:, :3] += sweep['sensor2lidar_translation']
                points_sweep[:, self.time_dim] = ts - sweep_ts
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)
        points = points[:, self.use_dim]
        results['points'] = points
        if self.translate2ego:
            lidar2lidarego = np.eye(4, dtype=np.float32)
            lidar2lidarego[:3, :3] = Quaternion(
            results['curr']['lidar2ego_rotation']).rotation_matrix
            lidar2lidarego[:3, 3] = results['curr']['lidar2ego_translation']
            lidar2lidarego = to_tensor(lidar2lidarego)
            results['points'].tensor[:, :3]  = results['points'].tensor[:, :3].matmul(lidar2lidarego[:3, :3].T) + lidar2lidarego[:3, 3]
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'


@PIPELINES.register_module()
class PointsFromLidartoEgo(object):
    
    def __init__(self, translate2ego=True, ego_cam='CAM_FRONT'):
        self.ego_cam=ego_cam
        self.translate2ego = translate2ego

    def __call__(self, results):
        if self.translate2ego:
            # lidar2lidarego = np.eye(4, dtype=np.float32)
            # lidar2lidarego[:3, :3] = Quaternion(
            # results['curr']['lidar2ego_rotation']).rotation_matrix
            # lidar2lidarego[:3, 3] = results['curr']['lidar2ego_translation']
            # lidar2lidarego = to_tensor(lidar2lidarego)
            # results['points'].tensor[:, :3]  = results['points'].tensor[:, :3].matmul(lidar2lidarego[:3, :3].T) + lidar2lidarego[:3, 3]

            lidar2lidarego = np.eye(4, dtype=np.float32)
            lidar2lidarego[:3, :3] = Quaternion(
                results['curr']['lidar2ego_rotation']).rotation_matrix
            lidar2lidarego[:3, 3] = results['curr']['lidar2ego_translation']

            lidarego2global = np.eye(4, dtype=np.float32)
            lidarego2global[:3, :3] = Quaternion(
                results['curr']['ego2global_rotation']).rotation_matrix
            lidarego2global[:3, 3] = results['curr']['ego2global_translation']

            camego2global = np.eye(4, dtype=np.float32)
            camego2global[:3, :3] = Quaternion(
                results['curr']['cams'][self.ego_cam]
                ['ego2global_rotation']).rotation_matrix
            camego2global[:3, 3] = results['curr']['cams'][self.ego_cam][
                'ego2global_translation']
            lidar2camego = np.linalg.inv(camego2global) @ lidarego2global @ lidar2lidarego
            lidar2camego = to_tensor(lidar2camego)
            results['points'].tensor[:, :3]  = results['points'].tensor[:, :3].matmul(lidar2camego[:3, :3].T) + lidar2camego[:3, 3]

        return results


@PIPELINES.register_module()
class PointSegClassMapping(object):
    """Map original semantic class to valid category ids.

    Map valid classes as 0~len(valid_cat_ids)-1 and
    others as len(valid_cat_ids).

    Args:
        valid_cat_ids (tuple[int]): A tuple of valid category.
        max_cat_id (int, optional): The max possible cat_id in input
            segmentation mask. Defaults to 40.
    """

    def __init__(self, valid_cat_ids, max_cat_id=40):
        assert max_cat_id >= np.max(valid_cat_ids), \
            'max_cat_id should be greater than maximum id in valid_cat_ids'

        self.valid_cat_ids = valid_cat_ids
        self.max_cat_id = int(max_cat_id)

        # build cat_id to class index mapping
        neg_cls = len(valid_cat_ids)
        self.cat_id2class = np.ones(
            self.max_cat_id + 1, dtype=np.int) * neg_cls
        for cls_idx, cat_id in enumerate(valid_cat_ids):
            self.cat_id2class[cat_id] = cls_idx

    def __call__(self, results):
        """Call function to map original semantic class to valid category ids.

        Args:
            results (dict): Result dict containing point semantic masks.

        Returns:
            dict: The result dict containing the mapped category ids.
                Updated key and value are described below.

                - pts_semantic_mask (np.ndarray): Mapped semantic masks.
        """
        assert 'pts_semantic_mask' in results
        pts_semantic_mask = results['pts_semantic_mask']

        converted_pts_sem_mask = self.cat_id2class[pts_semantic_mask]

        results['pts_semantic_mask'] = converted_pts_sem_mask
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(valid_cat_ids={self.valid_cat_ids}, '
        repr_str += f'max_cat_id={self.max_cat_id})'
        return repr_str


@PIPELINES.register_module()
class NormalizePointsColor(object):
    """Normalize color of points.

    Args:
        color_mean (list[float]): Mean color of the point cloud.
    """

    def __init__(self, color_mean):
        self.color_mean = color_mean

    def __call__(self, results):
        """Call function to normalize color of points.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the normalized points.
                Updated key and value are described below.

                - points (:obj:`BasePoints`): Points after color normalization.
        """
        points = results['points']
        assert points.attribute_dims is not None and \
            'color' in points.attribute_dims.keys(), \
            'Expect points have color attribute'
        if self.color_mean is not None:
            points.color = points.color - \
                points.color.new_tensor(self.color_mean)
        points.color = points.color / 255.0
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(color_mean={self.color_mean})'
        return repr_str


@PIPELINES.register_module()
class LoadPointsFromFile(object):
    """Load Points From File.

    Load points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int, optional): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int], optional): Which dimensions of the points to use.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool, optional): Whether to use shifted height.
            Defaults to False.
        use_color (bool, optional): Whether to use color features.
            Defaults to False.
        file_client_args (dict, optional): Config dict of file clients,
            refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(self,
                 coord_type,
                 load_dim=6,
                 use_dim=[0, 1, 2],
                 shift_height=False,
                 use_color=False,
                 dtype='float32',
                 file_client_args=dict(backend='disk'),
                 translate2ego=True,
                 ):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        if dtype=='float32':
            self.dtype = np.float32
        elif dtype== 'float16':
            self.dtype = np.float16
        else:
            assert False
        self.translate2ego = translate2ego

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=self.dtype)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=self.dtype)

        return points


    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_filename = results['pts_filename']
        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]



        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)

        results['points'] = points
        if self.translate2ego:
            lidar2lidarego = np.eye(4, dtype=np.float32)
            lidar2lidarego[:3, :3] = Quaternion(
            results['curr']['lidar2ego_rotation']).rotation_matrix
            lidar2lidarego[:3, 3] = results['curr']['lidar2ego_translation']
            lidar2lidarego = to_tensor(lidar2lidarego)
            results['points'].tensor[:, :3]  = results['points'].tensor[:, :3].matmul(lidar2lidarego[:3, :3].T) + lidar2lidarego[:3, 3]

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'file_client_args={self.file_client_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        return repr_str




@PIPELINES.register_module()
class LoadPointsFromDict(LoadPointsFromFile):
    """Load Points From Dict."""

    def __call__(self, results):
        assert 'points' in results
        return results


@PIPELINES.register_module()
class LoadAnnotations3D(LoadAnnotations):
    """Load Annotations3D.

    Load instance mask and semantic mask of points and
    encapsulate the items into related fields.

    Args:
        with_bbox_3d (bool, optional): Whether to load 3D boxes.
            Defaults to True.
        with_label_3d (bool, optional): Whether to load 3D labels.
            Defaults to True.
        with_attr_label (bool, optional): Whether to load attribute label.
            Defaults to False.
        with_mask_3d (bool, optional): Whether to load 3D instance masks.
            for points. Defaults to False.
        with_seg_3d (bool, optional): Whether to load 3D semantic masks.
            for points. Defaults to False.
        with_bbox (bool, optional): Whether to load 2D boxes.
            Defaults to False.
        with_label (bool, optional): Whether to load 2D labels.
            Defaults to False.
        with_mask (bool, optional): Whether to load 2D instance masks.
            Defaults to False.
        with_seg (bool, optional): Whether to load 2D semantic masks.
            Defaults to False.
        with_bbox_depth (bool, optional): Whether to load 2.5D boxes.
            Defaults to False.
        poly2mask (bool, optional): Whether to convert polygon annotations
            to bitmasks. Defaults to True.
        seg_3d_dtype (dtype, optional): Dtype of 3D semantic masks.
            Defaults to int64
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details.
    """

    def __init__(self,
                 with_bbox_3d=True,
                 with_label_3d=True,
                 with_attr_label=False,
                 with_mask_3d=False,
                 with_seg_3d=False,
                 with_bbox=False,
                 with_label=False,
                 with_mask=False,
                 with_seg=False,
                 with_bbox_depth=False,
                 poly2mask=True,
                 seg_3d_dtype=np.int64,
                 file_client_args=dict(backend='disk')):
        super().__init__(
            with_bbox,
            with_label,
            with_mask,
            with_seg,
            poly2mask,
            file_client_args=file_client_args)
        self.with_bbox_3d = with_bbox_3d
        self.with_bbox_depth = with_bbox_depth
        self.with_label_3d = with_label_3d
        self.with_attr_label = with_attr_label
        self.with_mask_3d = with_mask_3d
        self.with_seg_3d = with_seg_3d
        self.seg_3d_dtype = seg_3d_dtype

    def _load_bboxes_3d(self, results):
        """Private function to load 3D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box annotations.
        """
        results['gt_bboxes_3d'] = results['ann_infos'][0]
        results['bbox3d_fields'].append('gt_bboxes_3d')
        return results

    def _load_bboxes_depth(self, results):
        """Private function to load 2.5D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 2.5D bounding box annotations.
        """
        results['centers2d'] = results['ann_info']['centers2d']
        results['depths'] = results['ann_info']['depths']
        return results

    def _load_labels_3d(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results['gt_labels_3d'] = results['ann_infos'][1]
        return results

    def _load_attr_labels(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results['attr_labels'] = results['ann_infos']['attr_labels']
        return results

    def _load_masks_3d(self, results):
        """Private function to load 3D mask annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D mask annotations.
        """
        pts_instance_mask_path = results['ann_infos']['pts_instance_mask_path']

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            mask_bytes = self.file_client.get(pts_instance_mask_path)
            pts_instance_mask = np.frombuffer(mask_bytes, dtype=np.int64)
        except ConnectionError:
            mmcv.check_file_exist(pts_instance_mask_path)
            pts_instance_mask = np.fromfile(
                pts_instance_mask_path, dtype=np.int64)

        results['pts_instance_mask'] = pts_instance_mask
        results['pts_mask_fields'].append('pts_instance_mask')
        return results

    def _load_semantic_seg_3d(self, results):
        """Private function to load 3D semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing the semantic segmentation annotations.
        """
        pts_semantic_mask_path = results['ann_infos']['pts_semantic_mask_path']

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            mask_bytes = self.file_client.get(pts_semantic_mask_path)
            # add .copy() to fix read-only bug
            pts_semantic_mask = np.frombuffer(
                mask_bytes, dtype=self.seg_3d_dtype).copy()
        except ConnectionError:
            mmcv.check_file_exist(pts_semantic_mask_path)
            pts_semantic_mask = np.fromfile(
                pts_semantic_mask_path, dtype=np.int64)

        results['pts_semantic_mask'] = pts_semantic_mask
        results['pts_seg_fields'].append('pts_semantic_mask')
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box, label, mask and
                semantic segmentation annotations.
        """
        results = super().__call__(results)
        if self.with_bbox_3d:
            results = self._load_bboxes_3d(results)
            if results is None:
                return None
        if self.with_bbox_depth:
            results = self._load_bboxes_depth(results)
            if results is None:
                return None
        if self.with_label_3d:
            results = self._load_labels_3d(results)
        if self.with_attr_label:
            results = self._load_attr_labels(results)
        if self.with_mask_3d:
            results = self._load_masks_3d(results)
        if self.with_seg_3d:
            results = self._load_semantic_seg_3d(results)

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}with_bbox_3d={self.with_bbox_3d}, '
        repr_str += f'{indent_str}with_label_3d={self.with_label_3d}, '
        repr_str += f'{indent_str}with_attr_label={self.with_attr_label}, '
        repr_str += f'{indent_str}with_mask_3d={self.with_mask_3d}, '
        repr_str += f'{indent_str}with_seg_3d={self.with_seg_3d}, '
        repr_str += f'{indent_str}with_bbox={self.with_bbox}, '
        repr_str += f'{indent_str}with_label={self.with_label}, '
        repr_str += f'{indent_str}with_mask={self.with_mask}, '
        repr_str += f'{indent_str}with_seg={self.with_seg}, '
        repr_str += f'{indent_str}with_bbox_depth={self.with_bbox_depth}, '
        repr_str += f'{indent_str}poly2mask={self.poly2mask})'
        return repr_str


@PIPELINES.register_module()
class PointToMultiViewDepth(object):

    def __init__(self, grid_config, downsample=1):
        self.downsample = downsample
        self.grid_config = grid_config

    def points2depthmap(self, points, height, width):
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((height, width), dtype=torch.float32)
        coor = torch.round(points[:, :2] / self.downsample)
        depth = points[:, 2]
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                depth < self.grid_config['depth'][1]) & (
                    depth >= self.grid_config['depth'][0])
        coor, depth = coor[kept1], depth[kept1]

        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]


        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]


        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth
     
        return depth_map

    def __call__(self, results):
        points_lidar = results['points']
        imgs, rots, trans, intrins = results['img_inputs'][:4]
        post_rots, post_trans, bda = results['img_inputs'][4:]
        depth_map_list = []
        for cid in range(len(results['cam_names'])):
            cam_name = results['cam_names'][cid]
            # lidar2lidarego = np.eye(4, dtype=np.float32)
            # lidar2lidarego[:3, :3] = Quaternion(
            #     results['curr']['lidar2ego_rotation']).rotation_matrix
            # lidar2lidarego[:3, 3] = results['curr']['lidar2ego_translation']
            # lidar2lidarego = to_tensor(lidar2lidarego)

            lidarego2global = np.eye(4, dtype=np.float32)
            lidarego2global[:3, :3] = Quaternion(
                results['curr']['ego2global_rotation']).rotation_matrix
            lidarego2global[:3, 3] = results['curr']['ego2global_translation']
            lidarego2global = to_tensor(lidarego2global)

            cam2camego = np.eye(4, dtype=np.float32)
            cam2camego[:3, :3] = Quaternion(
                results['curr']['cams'][cam_name]
                ['sensor2ego_rotation']).rotation_matrix
            cam2camego[:3, 3] = results['curr']['cams'][cam_name][
                'sensor2ego_translation']
            cam2camego = to_tensor(cam2camego)

            camego2global = np.eye(4, dtype=np.float32)
            camego2global[:3, :3] = Quaternion(
                results['curr']['cams'][cam_name]
                ['ego2global_rotation']).rotation_matrix
            camego2global[:3, 3] = results['curr']['cams'][cam_name][
                'ego2global_translation']
            camego2global = to_tensor(camego2global)

            cam2img = np.eye(4, dtype=np.float32)
            cam2img = to_tensor(cam2img)
            cam2img[:3, :3] = intrins[cid]

            lidar2cam = torch.inverse(camego2global.matmul(cam2camego)).matmul(lidarego2global)
            # lidarego2global.matmul(lidar2lidarego))
            lidar2img = cam2img.matmul(lidar2cam)
            points_img = points_lidar.tensor[:, :3].matmul(
                lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
            points_img = torch.cat(
                [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]],
                1)
            points_img = points_img.matmul(
                post_rots[cid].T) + post_trans[cid:cid + 1, :]
            depth_map = self.points2depthmap(points_img, imgs.shape[2],
                                             imgs.shape[3])  
            depth_map_list.append(depth_map)
          
        depth_map = torch.stack(depth_map_list)

        results['gt_depth'] = depth_map

        return results


def mmlabNormalize(img, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True, debug=False):
    from mmcv.image.photometric import imnormalize
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    to_rgb = to_rgb
    if debug:
        print('warning, debug in mmlabNormalize')
        img = np.asarray(img) # not normalize for visualization
    else:
        img = imnormalize(np.array(img), mean, std, to_rgb)
    img = torch.tensor(img).float().permute(2, 0, 1).contiguous()
    return img


@PIPELINES.register_module()
class PrepareImageInputs(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(
        self,
        data_config,
        is_train=False,
        sequential=False,
        ego_cam='CAM_FRONT',
        img_corruptions=None,
        normalize_cfg=dict(
             mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True, debug=False
        )
    ):
        self.is_train = is_train
        self.data_config = data_config
        self.normalize_img = mmlabNormalize
        self.sequential = sequential
        self.ego_cam = ego_cam
        self.normalize_cfg = normalize_cfg
        self.img_corruptions = img_corruptions

    def get_rot(self, h):
        return torch.Tensor(
            np.array([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
            ]))

    def img_transform(self, img, post_rot, post_tran, resize, resize_dims,
                      crop, flip, rotate):
        # adjust image
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)

        # post-homography transformation
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, post_rot, post_tran

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        return img

    def choose_cams(self):
        if self.is_train and self.data_config['Ncams'] < len(
                self.data_config['cams']):
            cam_names = np.random.choice(
                self.data_config['cams'],
                self.data_config['Ncams'],
                replace=False)
        else:
            cam_names = self.data_config['cams']
        return cam_names

    def sample_augmentation(self, H, W, flip=None, scale=None):
        fH, fW = self.data_config['input_size']
        H, W = self.data_config['src_size']
        if self.is_train:
            # resize = float(fW) / float(W)
            # resize += np.random.uniform(*self.data_config['resize'])
            resize = np.random.uniform(*self.data_config["resize"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_config['crop_h'])) *
                         newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.data_config['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config['rot'])
        else:
            # resize = float(fW) / float(W)
            resize = max(fH / H, fW / W)
            # resize += self.data_config.get('resize_test', 0.0)
            if scale is not None:
                resize = scale
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def get_sensor2ego_transformation(self,
                                      cam_info,
                                      key_info,
                                      cam_name,
                                      ego_cam=None):
        if ego_cam is None:
            ego_cam = cam_name
        w, x, y, z = cam_info['cams'][cam_name]['sensor2ego_rotation']
        # sweep sensor to sweep ego
        sweepsensor2sweepego_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        sweepsensor2sweepego_tran = torch.Tensor(
            cam_info['cams'][cam_name]['sensor2ego_translation'])
        sweepsensor2sweepego = sweepsensor2sweepego_rot.new_zeros((4, 4))
        sweepsensor2sweepego[3, 3] = 1
        sweepsensor2sweepego[:3, :3] = sweepsensor2sweepego_rot
        sweepsensor2sweepego[:3, -1] = sweepsensor2sweepego_tran
        # sweep ego to global
        w, x, y, z = cam_info['cams'][cam_name]['ego2global_rotation']
        sweepego2global_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        sweepego2global_tran = torch.Tensor(
            cam_info['cams'][cam_name]['ego2global_translation'])
        sweepego2global = sweepego2global_rot.new_zeros((4, 4))
        sweepego2global[3, 3] = 1
        sweepego2global[:3, :3] = sweepego2global_rot
        sweepego2global[:3, -1] = sweepego2global_tran

        # global sensor to cur ego
        w, x, y, z = key_info['cams'][ego_cam]['ego2global_rotation']
        keyego2global_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        keyego2global_tran = torch.Tensor(
            key_info['cams'][ego_cam]['ego2global_translation'])
        keyego2global = keyego2global_rot.new_zeros((4, 4))
        keyego2global[3, 3] = 1
        keyego2global[:3, :3] = keyego2global_rot
        keyego2global[:3, -1] = keyego2global_tran
        global2keyego = keyego2global.inverse()

        sweepsensor2keyego = \
            global2keyego @ sweepego2global @ sweepsensor2sweepego

        # global sensor to cur ego
        w, x, y, z = key_info['cams'][cam_name]['ego2global_rotation']
        keyego2global_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        keyego2global_tran = torch.Tensor(
            key_info['cams'][cam_name]['ego2global_translation'])
        keyego2global = keyego2global_rot.new_zeros((4, 4))
        keyego2global[3, 3] = 1
        keyego2global[:3, :3] = keyego2global_rot
        keyego2global[:3, -1] = keyego2global_tran
        global2keyego = keyego2global.inverse()

        # cur ego to sensor
        w, x, y, z = key_info['cams'][cam_name]['sensor2ego_rotation']
        keysensor2keyego_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        keysensor2keyego_tran = torch.Tensor(
            key_info['cams'][cam_name]['sensor2ego_translation'])
        keysensor2keyego = keysensor2keyego_rot.new_zeros((4, 4))
        keysensor2keyego[3, 3] = 1
        keysensor2keyego[:3, :3] = keysensor2keyego_rot
        keysensor2keyego[:3, -1] = keysensor2keyego_tran
        keyego2keysensor = keysensor2keyego.inverse()
        keysensor2sweepsensor = (
            keyego2keysensor @ global2keyego @ sweepego2global
            @ sweepsensor2sweepego).inverse()
        return sweepsensor2keyego, keysensor2sweepsensor


    def get_sensor_transforms(self, cam_info, cam_name):
        w, x, y, z = cam_info['cams'][cam_name]['sensor2ego_rotation']
        # sweep sensor to sweep ego
        sensor2ego_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        sensor2ego_tran = torch.Tensor(
            cam_info['cams'][cam_name]['sensor2ego_translation'])
        sensor2ego = sensor2ego_rot.new_zeros((4, 4))
        sensor2ego[3, 3] = 1
        sensor2ego[:3, :3] = sensor2ego_rot
        sensor2ego[:3, -1] = sensor2ego_tran
        # sweep ego to global
        w, x, y, z = cam_info['cams'][cam_name]['ego2global_rotation']
        ego2global_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        ego2global_tran = torch.Tensor(
            cam_info['cams'][cam_name]['ego2global_translation'])
        ego2global = ego2global_rot.new_zeros((4, 4))
        ego2global[3, 3] = 1
        ego2global[:3, :3] = ego2global_rot
        ego2global[:3, -1] = ego2global_tran
        return sensor2ego, ego2global

    def get_inputs(self, results, scale=None):
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []
        sensor2egos = []
        ego2globals = []
        cam_names = self.choose_cams()
        results['cam_names'] = cam_names
        results['input_size'] = self.data_config['input_size']
        canvas = []
        sensor2sensors = []
        results['img_augs'] = {}
        for cam_name in cam_names:
            cam_data = results['curr']['cams'][cam_name]
            filename = cam_data['data_path']
            if self.img_corruptions in ['sun', 'noise', 'rain', 'snow', 'fog']:
                filename = filename.split('/')
                filename[2] = 'nuscenes_aug'
                filename[3] = f'samples_{self.img_corruptions}'
                filename = osp.join(*filename)
            
            img = Image.open(filename)
            


            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            intrin = torch.Tensor(cam_data['cam_intrinsic'])

            sensor2keyego, sensor2sensor = \
                self.get_sensor2ego_transformation(results['curr'],
                                                   results['curr'],
                                                   cam_name,
                                                   self.ego_cam)
            rot = sensor2keyego[:3, :3]
            tran = sensor2keyego[:3, 3]
            sensor2ego, ego2global = \
                self.get_sensor_transforms(results['curr'], cam_name)
            # image view augmentation (resize, crop, horizontal flip, rotate)
            if results.get('tta_config', None) is not None:
                flip = results['tta_config']['tta_flip']
            else: flip = None
            img_augs = self.sample_augmentation(
                H=img.height, W=img.width, flip=flip, scale=scale)
            resize, resize_dims, crop, flip, rotate = img_augs
            results['img_augs'][cam_name] = img_augs
            img, post_rot2, post_tran2 = \
                self.img_transform(img, post_rot,
                                   post_tran,
                                   resize=resize,
                                   resize_dims=resize_dims,
                                   crop=crop,
                                   flip=flip,
                                   rotate=rotate)

            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            canvas.append(np.array(img))
            if self.img_corruptions == 'drop':
                imgs.append(self.normalize_img(img, **self.normalize_cfg)* 0)
            else:
                imgs.append(self.normalize_img(img, **self.normalize_cfg))

            if self.sequential:
                assert 'adjacent' in results
                for adj_info in results['adjacent']:
                    filename_adj = adj_info['cams'][cam_name]['data_path']
                    img_adjacent = Image.open(filename_adj)
                    img_adjacent = self.img_transform_core(
                        img_adjacent,
                        resize_dims=resize_dims,
                        crop=crop,
                        flip=flip,
                        rotate=rotate)
                    imgs.append(self.normalize_img(img_adjacent, **self.normalize_cfg))
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)
            sensor2sensors.append(sensor2sensor)
            sensor2egos.append(sensor2ego)
            ego2globals.append(ego2global)

        if self.sequential:
            for adj_info in results['adjacent']:
                post_trans.extend(post_trans[:len(cam_names)])
                post_rots.extend(post_rots[:len(cam_names)])
                intrins.extend(intrins[:len(cam_names)])

                # align
                trans_adj = []
                rots_adj = []
                sensor2sensors_adj = []
                for cam_name in cam_names:
                    adjsensor2keyego, sensor2sensor = \
                        self.get_sensor2ego_transformation(adj_info,
                                                           results['curr'],
                                                           cam_name,
                                                           self.ego_cam)
                    rot = adjsensor2keyego[:3, :3]
                    tran = adjsensor2keyego[:3, 3]
                    rots_adj.append(rot)
                    trans_adj.append(tran)
                    sensor2sensors_adj.append(sensor2sensor)
                for cam_name in cam_names:
                    sensor2ego, ego2global = \
                        self.get_sensor_transforms(adj_info, cam_name)
                    sensor2egos.append(sensor2ego)
                    ego2globals.append(ego2global)

                rots.extend(rots_adj)
                trans.extend(trans_adj)
                sensor2sensors.extend(sensor2sensors_adj)
        imgs = torch.stack(imgs)
        
        sensor2egos = torch.stack(sensor2egos)
        ego2globals = torch.stack(ego2globals)

        rots = torch.stack(rots)
        trans = torch.stack(trans)
        intrins = torch.stack(intrins)
        post_rots = torch.stack(post_rots)
        post_trans = torch.stack(post_trans)
        sensor2sensors = torch.stack(sensor2sensors)
        results['canvas'] = canvas
        results['sensor2sensors'] = sensor2sensors
        return (imgs, rots, trans, intrins, post_rots, post_trans), (sensor2egos, ego2globals)

    def __call__(self, results):
        results['img_inputs'], results['aux_cam_params'] = self.get_inputs(results)
        return results


@PIPELINES.register_module()
class LoadAnnotationsBEVDepth(object):


    def __init__(self, bda_aug_conf, classes, with_2d_bbox=False, with_ego_as_agent=False, is_train=True):
        self.bda_aug_conf = bda_aug_conf
        self.is_train = is_train
        self.classes = classes
        self.with_2d_bbox = with_2d_bbox
        self.min_size = 2.0
        self.with_ego_as_agent = with_ego_as_agent

    def sample_bda_augmentation(self, tta_config=None):
        """Generate bda augmentation values based on bda_config."""
        if self.is_train:
            rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
            scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
            flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
            flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
            translation_std = self.bda_aug_conf.get('tran_lim', [0.0, 0.0, 0.0])
            tran_bda = np.random.normal(scale=translation_std, size=3).T
        else:
            rotate_bda = 0
            scale_bda = 1.0
            if tta_config is not None:
                flip_dx = tta_config['flip_dx']
                flip_dy = tta_config['flip_dy']
            else:
                flip_dx = False
                flip_dy = False
            tran_bda = np.zeros((1, 3), dtype=np.float32)
        return rotate_bda, scale_bda, flip_dx, flip_dy, tran_bda


    def bev_transform(self, gt_boxes, rotate_angle, scale_ratio, flip_dx,
                      flip_dy):
        rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
        rot_sin = torch.sin(rotate_angle)
        rot_cos = torch.cos(rotate_angle)
        rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                                [0, 0, 1]])
        scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                                  [0, 0, scale_ratio]])
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dx:
            flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0],
                                                [0, 0, 1]])
        if flip_dy:
            flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0],
                                                [0, 0, 1]])
        rot_mat = flip_mat @ (scale_mat @ rot_mat)
        if gt_boxes.shape[0] > 0:
            gt_boxes[:, :3] = (
                rot_mat @ gt_boxes[:, :3].unsqueeze(-1)).squeeze(-1)
            gt_boxes[:, 3:6] *= scale_ratio
            gt_boxes[:, 6] += rotate_angle
            if flip_dx:
                gt_boxes[:,
                         6] = 2 * torch.asin(torch.tensor(1.0)) - gt_boxes[:,
                                                                           6]
            if flip_dy:
                gt_boxes[:, 6] = -gt_boxes[:, 6]
            gt_boxes[:, 7:] = (
                rot_mat[:2, :2] @ gt_boxes[:, 7:].unsqueeze(-1)).squeeze(-1)
        return gt_boxes, rot_mat

    def _bboxes_transform(self, bboxes, centers2d, gt_labels, depths, resize, crop, flip, fH, fW):
        assert len(bboxes) == len(centers2d) == len(gt_labels) == len(depths)
        bboxes = bboxes * resize
        bboxes[:, 0] = bboxes[:, 0] - crop[0]
        bboxes[:, 1] = bboxes[:, 1] - crop[1]
        bboxes[:, 2] = bboxes[:, 2] - crop[0]
        bboxes[:, 3] = bboxes[:, 3] - crop[1]
        bboxes[:, 0] = np.clip(bboxes[:, 0], 0, fW)
        bboxes[:, 2] = np.clip(bboxes[:, 2], 0, fW)
        bboxes[:, 1] = np.clip(bboxes[:, 1], 0, fH) 
        bboxes[:, 3] = np.clip(bboxes[:, 3], 0, fH)
        keep = ((bboxes[:, 2] - bboxes[:, 0]) >= self.min_size) & ((bboxes[:, 3] - bboxes[:, 1]) >= self.min_size)


        if flip:
            x0 = bboxes[:, 0].copy()
            x1 = bboxes[:, 2].copy()
            bboxes[:, 2] = fW - x0
            bboxes[:, 0] = fW - x1
        bboxes = bboxes[keep]

        centers2d  = centers2d * resize
        centers2d[:, 0] = centers2d[:, 0] - crop[0]
        centers2d[:, 1] = centers2d[:, 1] - crop[1]
        centers2d[:, 0] = np.clip(centers2d[:, 0], 0, fW)
        centers2d[:, 1] = np.clip(centers2d[:, 1], 0, fH) 
        if flip:
            centers2d[:, 0] = fW - centers2d[:, 0]

        centers2d = centers2d[keep]
        gt_labels = gt_labels[keep]
        depths = depths[keep]

        return bboxes, centers2d, gt_labels, depths


    def _filter_invisible(self, bboxes, centers2d, gt_labels, depths, fH, fW ):
        # filter invisible 2d bboxes
        assert len(bboxes) == len(centers2d) == len(gt_labels) == len(depths)

        indices_maps = np.zeros((fH,fW))
        tmp_bboxes = np.zeros_like(bboxes)
        tmp_bboxes[:, :2] = np.ceil(bboxes[:, :2])
        tmp_bboxes[:, 2:] = np.floor(bboxes[:, 2:])
        tmp_bboxes = tmp_bboxes.astype(np.int64)
        sort_idx = np.argsort(-depths, axis=0, kind='stable')
        tmp_bboxes = tmp_bboxes[sort_idx]
        bboxes = bboxes[sort_idx]
        depths = depths[sort_idx]
        centers2d = centers2d[sort_idx]
        gt_labels = gt_labels[sort_idx]
        for i in range(bboxes.shape[0]):
            u1, v1, u2, v2 = tmp_bboxes[i]
            indices_maps[v1:v2, u1:u2] = i
        indices_res = np.unique(indices_maps).astype(np.int64)
        bboxes = bboxes[indices_res]
        depths = depths[indices_res]
        centers2d = centers2d[indices_res]
        gt_labels = gt_labels[indices_res]

        return bboxes, centers2d, gt_labels, depths

    def __call__(self, results):
        gt_boxes, gt_labels = results['ann_infos']['gt_boxes_3d'], results['ann_infos']['gt_labels_3d']

        if self.with_ego_as_agent:
            ego_xyz = np.array([0, 0, 0])
            ego_wlh = np.array([4.084, 1.85, 1.8])
            ego_yaw = np.array([0])
            ego_vel = results['curr']['gt_ego_lcf_feat'][:2]
            ego_box = np.concatenate([ego_xyz, ego_wlh, ego_yaw, ego_vel])
            gt_boxes =  [ego_box] + gt_boxes
            gt_labels = [0] + gt_labels

            if 'instance_inds' in results.keys():
                results['instance_inds'] = np.concatenate([[1e7], results['instance_inds']])

        if self.with_2d_bbox:
            # gt_boxes_2d, gt_labels_2d = results['ann_infos']['gt_boxes_2d'], results['ann_infos']['gt_labels_2d']
            # gt_centers2d, gt_depth2d = results['ann_infos']['centers2d'], results['ann_infos']['depths']
            new_gt_bboxes = []
            new_centers2d = []
            new_gt_labels = []
            new_depths = []
            fH, fW = results['input_size']
            for cam in results['cam_names']:
                camera_types_2d = [
                    'CAM_FRONT',
                    'CAM_FRONT_RIGHT',
                    'CAM_FRONT_LEFT',
                    'CAM_BACK',
                    'CAM_BACK_LEFT',
                    'CAM_BACK_RIGHT',
                ]
                i = camera_types_2d.index(cam)
                resize, resize_dims, crop, flip, rotate = results['img_augs'][cam]
                gt_bboxes_2d = results['ann_infos']['gt_boxes_2d'][i]
                centers2d = results['ann_infos']['centers2d'][i]
                gt_labels_2d = results['ann_infos']['gt_labels_2d'][i]
                depths = results['ann_infos']['depths'][i]
                if len(gt_bboxes_2d) != 0:
                    gt_bboxes_2d, centers2d, gt_labels_2d, depths = self._bboxes_transform(
                        gt_bboxes_2d, 
                        centers2d,
                        gt_labels_2d,
                        depths,
                        resize=resize,
                        crop=crop,
                        flip=flip,
                        fH=fH,
                        fW=fW,
                    )
                if len(gt_bboxes_2d) != 0:
                    gt_bboxes_2d, centers2d, gt_labels_2d, depths =  self._filter_invisible(gt_bboxes_2d, centers2d, gt_labels_2d, depths, fH, fW)

                new_gt_bboxes.append(to_tensor(gt_bboxes_2d))
                new_centers2d.append(to_tensor(centers2d))
                new_gt_labels.append(to_tensor(gt_labels_2d))
                new_depths.append(to_tensor(depths))

            results['gt_bboxes_2d'] = new_gt_bboxes
            results['centers2d'] = new_centers2d
            results['gt_labels_2d'] = new_gt_labels
            results['depths2d'] = new_depths

        gt_boxes, gt_labels = torch.Tensor(np.array(gt_boxes)), torch.tensor(np.array(gt_labels))
        tta_confg = results.get('tta_config', None)

        rotate_bda, scale_bda, flip_dx, flip_dy, tran_bda = self.sample_bda_augmentation(tta_confg)

        bda_mat = torch.zeros(4, 4)
        bda_mat[3, 3] = 1
        gt_boxes, bda_rot = self.bev_transform(gt_boxes, rotate_bda, scale_bda,
                                               flip_dx, flip_dy)
        if 'points' in results:
            points = results['points'].tensor
            points_aug = (bda_rot @ points[:, :3].unsqueeze(-1)).squeeze(-1)
            points[:, :3] = points_aug + tran_bda
            points = results['points'].new_point(points)
            results['points'] = points
        
        bda_mat[:3, :3] = bda_rot
        if len(gt_boxes) == 0:
            gt_boxes = torch.zeros(0, 9)
        results['gt_bboxes_3d'] = \
            LiDARInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1],
                                 origin=(0.5, 0.5, 0.5))
        results['gt_labels_3d'] = gt_labels
        imgs, rots, trans, intrins = results['img_inputs'][:4]
        post_rots, post_trans = results['img_inputs'][4:]
        results['img_inputs'] = (imgs, rots, trans, intrins, post_rots,
                                 post_trans, bda_rot)
        
        results['flip_dx'] = flip_dx
        results['flip_dy'] = flip_dy
        results['rotate_bda'] = rotate_bda
        results['scale_bda'] = scale_bda
        results['bda_mat'] = bda_mat
        if 'ego_pose' in results:
            results['ori_ego_pose'] = results['ego_pose'].clone()
            results['ego_pose'] =  results['ego_pose'] @ torch.inverse(bda_mat)
            results['ego_pose_inv'] = bda_mat @ results['ego_pose_inv']
        return results

