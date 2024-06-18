# Copyright (c) 2023-2024, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# TODO: add license here


import torch
import torch.nn.functional as F
import torch.nn as nn
from mmcv.runner import force_fp32
import os
from mmdet3d.ops.bev_pool_v2.bev_pool import TRTBEVPoolv2
from mmdet.models import DETECTORS
from mmdet3d.models import builder
from mmdet3d.models.detectors import CenterPoint
from mmdet3d.models.builder import build_head, build_neck
import numpy as np
import copy 
import spconv.pytorch as spconv
from tqdm import tqdm 
from mmdet3d.models.fbbev.utils import run_time
import torch
from torchvision.utils import make_grid
import torchvision
import matplotlib
import matplotlib.pyplot as plt
import cv2
from collections import defaultdict
from mmcv.runner import get_dist_info
from mmdet.core import reduce_mean
import mmcv
from mmdet3d.datasets.utils import nuscenes_get_rt_matrix
from mmdet3d.core.bbox import box_np_ops # , corner_to_surfaces_3d, points_in_convex_polygon_3d_jit
import gc
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
import pickle
import numpy as np
import math

import copy
import math
from mmcv.runner.base_module import BaseModule
from mmdet3d.models.detectors.base import Base3DDetector



import torch
import torch.nn as nn 
from mmcv.cnn import Linear, bias_init_with_prob, Scale

from mmcv.runner import force_fp32
from mmdet.core import (build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet.models.utils import build_transformer
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet3d.core.bbox.coders import build_bbox_coder
from ..streampetr.streampetr_utils import *
import copy
from mmdet.models.utils import NormedLinear
from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from mmdet3d.models.fbbev.utils import save_tensor
from mmcv.runner.base_module import BaseModule
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from .metric_stp3 import PlanningMetric
# from memory_profiler import profile
from matplotlib.backends.backend_agg import FigureCanvasAgg
import PIL.Image as Image




def get_ego_pos(points, pc_range):
    if points.size(-1) == 3:
        points = points * (pc_range[3:6] - pc_range[0:3]) + pc_range[0:3]
    elif  points.size(-1) == 2:
        points = points * (pc_range[3:5] - pc_range[0:2]) + pc_range[0:2]
    return points

def get_rel_pos(points, pc_range):
    if points.size(-1) == 3:
        return (points - pc_range[0:3]) / (pc_range[3:6] - pc_range[0:3])
    elif  points.size(-1) == 2:
        return (points - pc_range[0:2]) / (pc_range[3:5] - pc_range[0:2])


@HEADS.register_module()
class AD_MLP(Base3DDetector):
    """Implements the DETR transformer head.
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        num_classes (int): Number of categories excluding the background.
        in_channels (int): Number of channels in the input feature map.
        num_query (int): Number of query in Transformer.
        num_reg_fcs (int, optional): Number of fully-connected layers used in
            `FFN`, which is then used for the regression head. Default 2.
        transformer (obj:`mmcv.ConfigDict`|dict): Config for transformer.
            Default: None.
        sync_cls_avg_factor (bool): Whether to sync the avg_factor of
            all ranks. Default to False.
        positional_encoding (obj:`mmcv.ConfigDict`|dict):
            Config for position encoding.
        loss_cls (obj:`mmcv.ConfigDict`|dict): Config of the
            classification loss. Default `CrossEntropyLoss`.
        loss_bbox (obj:`mmcv.ConfigDict`|dict): Config of the
            regression loss. Default `L1Loss`.
        loss_iou (obj:`mmcv.ConfigDict`|dict): Config of the
            regression iou loss. Default `GIoULoss`.
        tran_cfg (obj:`mmcv.ConfigDict`|dict): Training config of
            transformer head.
        test_cfg (obj:`mmcv.ConfigDict`|dict): Testing config of
            transformer head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """
    _version = 2

    def __init__(self,
                 in_channels=256,
                 embed_dims=256,
                 num_query=1,
                 num_reg_fcs=2,
                 memory_len=12,
                 transformer=None,
                 sync_cls_avg_factor=False,
                 code_weights=None,
                 init_cfg=None,
                 point_cloud_range=None,
                 loss_plan_reg=dict(type='L1Loss', loss_weight=5.0),
                **kwargs):

        super().__init__()
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 2



        self.num_query = num_query
        self.in_channels = in_channels
        self.num_reg_fcs = num_reg_fcs
        # self.train_cfg = train_cfg
        # self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.embed_dims = embed_dims
        self.num_motion_mode = 6
        self.fut_steps = 6
        self.memory_len = 6
        self.ego_fut_mode = 3

       
        # self.code_weights = nn.Parameter(torch.tensor(
        #     self.code_weights), requires_grad=False)
        self.pc_range = nn.Parameter(torch.tensor(
            point_cloud_range), requires_grad=False)



        self.loss_plan_reg = build_loss(loss_plan_reg)

        # self.ego_map_decoder = build_transformer_layer_sequence(self.ego_map_decoder)
        # self.ego_decoder = build_transformer_layer_sequence(ego_agent_decoder)
        self._init_layers()
    
        self.planning_metric = PlanningMetric()
        self.count = 0
        # dummy
        self.history_sweep_time = None
        self.history_bev = None
        self.history_bev_before_encoder = None
        self.history_seq_ids = None
        self.history_forward_augs = None

    def _init_layers(self):
        """Initialize layers of the transformer head."""

        ego_fut_decoder = []
        ego_fut_dec_in_dim = self.embed_dims*2
        for i in range(self.num_reg_fcs):
            if i == 0:
                ego_fut_decoder.append(Linear(12, ego_fut_dec_in_dim))
            else:
                ego_fut_decoder.append(Linear(ego_fut_dec_in_dim, ego_fut_dec_in_dim))
            ego_fut_decoder.append(nn.ReLU())
        ego_fut_decoder.append(Linear(ego_fut_dec_in_dim, self.ego_fut_mode*self.fut_steps*2))
        self.ego_fut_decoder = nn.Sequential(*ego_fut_decoder)



    def forward_train(self,  img_metas=None, **kwargs):
        
        """
        NOTE: if I do not `detach` the tensor but use `clone`, there will be a CPU memory leak. I do not figure it out yet.
        """
        preds_plan_dicts = self.inner_forward(img_metas, **kwargs)
        return self.loss(
            preds_plan_dicts=preds_plan_dicts,
            img_metas=img_metas,
            **kwargs
        )



    def inner_forward(self,  img_metas=None, **kwargs):

        """
        NOTE: if I do not `detach` the tensor but use `clone`, there will be a CPU memory leak. I do not figure it out yet.
        """

        gt_ego_lcf_feat = torch.stack(kwargs['gt_ego_lcf_feat'], 0)
        gt_ego_fut_cmd = torch.stack(kwargs['gt_ego_fut_cmd'], 0)
        # gt_ego_fut_trajs =  torch.stack(kwargs['gt_ego_fut_trajs'], 0)
        self.ego_fut_steps = 6

        vel = gt_ego_lcf_feat[:, :2].unsqueeze(1).repeat(1, self.ego_fut_steps, 1) # * torch.arange(1, self.ego_fut_steps+1)
        accelation =  gt_ego_lcf_feat[:, 2:4].unsqueeze(1).repeat(1, self.ego_fut_steps, 1) * torch.arange(1, self.ego_fut_steps+1)[None, :, None].to(vel.device) * 0.5
        vel = vel # + accelation

        fut_traj_from_velo = torch.cumsum(vel * 0.5, 1)# [0]
        gt_ego_fut_trajs = kwargs['gt_ego_fut_trajs']# [0]
        # np.corrco(fut_traj_from_velo.cpu().numpy(), gt_ego_fut_trajs.cpu().numpy())

        input = torch.cat([gt_ego_lcf_feat, gt_ego_fut_cmd], -1)
        
        outputs_ego_trajs = self.ego_fut_decoder(input)
        # reference = inverse_sigmoid(reference_points.clone())
        outputs_ego_trajs = outputs_ego_trajs.reshape(outputs_ego_trajs.shape[0], 
                                                      self.ego_fut_mode, self.fut_steps, 2)
        start_of_sequence = torch.FloatTensor([
            single_img_metas['start_of_sequence'] 
            for single_img_metas in img_metas]).to(gt_ego_lcf_feat.device)

        timestamp = torch.FloatTensor([
            single_img_metas['timestamp'] 
            for single_img_metas in img_metas]).to(gt_ego_lcf_feat.device)

        ego_pose_inv = torch.stack([
            single_img_metas['ego_pose_inv'] 
            for single_img_metas in img_metas], 0).to(gt_ego_lcf_feat.device)

        ego_pose = torch.stack([
            single_img_metas['ego_pose'] 
            for single_img_metas in img_metas], 0).to(gt_ego_lcf_feat.device)

        data = dict(
            start_of_sequence = start_of_sequence,
            timestamp = timestamp,
            ego_pose_inv = ego_pose_inv,
            ego_pose = ego_pose,
        )

        preds_plan_dicts =  dict(
            # init_traj=reference_points[..., :2],
            data= data,
            ego_fut_preds=outputs_ego_trajs,
            # ego_trajs_in_global = ego_trajs_in_global,
            fut_traj_from_velo=fut_traj_from_velo
        )
        return preds_plan_dicts

    def forward_test(self, **kwargs):
        for key in ['img_metas', 'gt_ego_lcf_feat', 'gt_ego_fut_cmd', 'gt_ego_fut_trajs', 'gt_ego_fut_masks','gt_fut_segmentations', 'vad_ego_fut_trajs', 'gt_fut_segmentations_plus']:
            kwargs[key] = kwargs[key][0] 
        
        # img_metas = img_metas[0]
        return self.simple_test(**kwargs)
        

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_ego_fut_trajs=None,
             gt_ego_fut_cmd=None,
             gt_ego_fut_masks=None,
             preds_plan_dicts=None,
             img_metas=None,
             **kwargs,
            ):
        
        ego_fut_preds = preds_plan_dicts['ego_fut_preds']
        gt_ego_fut_trajs = torch.stack(gt_ego_fut_trajs)
        gt_ego_fut_cmd = torch.stack(gt_ego_fut_cmd)
        gt_ego_fut_masks = torch.stack(gt_ego_fut_masks)
        gt_ego_fut_trajs = torch.cat([gt_ego_fut_trajs[:,:1], (gt_ego_fut_trajs[:,1:] - gt_ego_fut_trajs[:,:-1])], 1)
        gt_ego_fut_trajs = gt_ego_fut_trajs.unsqueeze(1).repeat(1, self.ego_fut_mode, 1, 1)

        loss_plan_l1_weight = gt_ego_fut_cmd[..., None, None] * gt_ego_fut_masks[:, None, :, None]
        loss_plan_l1_weight = loss_plan_l1_weight.repeat(1, 1, 1, 2)
        
        loss_plan_l1 = self.loss_plan_reg(
            ego_fut_preds,
            gt_ego_fut_trajs,
            loss_plan_l1_weight
        )

        loss_plan_l1 = torch.nan_to_num(loss_plan_l1)
        loss_plan_dict = dict()
        loss_plan_dict['loss_plan_reg'] = loss_plan_l1

        return loss_plan_dict
    def aug_test(self): pass

    @force_fp32(apply_to=('reference_points', 'cam_params'))
    def point_sampling(self, reference_points, cam_params=None):

        rots, trans, intrins, post_rots, post_trans, bda = cam_params
        B, N, _ = trans.shape
        eps = 1e-5
        ogfH, ogfW = 900, 1600
        reference_points = reference_points[None, None].repeat(B, N, 1, 1, 1, 1)
        reference_points = torch.inverse(bda).view(B, 1, 1, 1, 1, 3,
                          3).matmul(reference_points.unsqueeze(-1)).squeeze(-1)
        reference_points -= trans.view(B, N, 1, 1, 1, 3)
        combine = rots.matmul(torch.inverse(intrins)).inverse()
        reference_points_cam = combine.view(B, N, 1, 1, 1, 3, 3).matmul(reference_points.unsqueeze(-1)).squeeze(-1)
        reference_points_cam = torch.cat([reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3])*eps),  reference_points_cam[..., 2:3]], 5
            )
        reference_points_cam = post_rots.view(B, N, 1, 1, 1, 3, 3).matmul(reference_points_cam.unsqueeze(-1)).squeeze(-1)
        reference_points_cam += post_trans.view(B, N, 1, 1, 1, 3) 
        # reference_points_cam[..., 0] /= ogfW
        # reference_points_cam[..., 1] /= ogfH
        mask = (reference_points_cam[..., 2:3] > eps)
        mask = (mask & (reference_points_cam[..., 0:1] > eps) 
                 & (reference_points_cam[..., 0:1] < (1.0-eps) * ogfW) 
                 & (reference_points_cam[..., 1:2] > eps) 
                 & (reference_points_cam[..., 1:2] < (1.0-eps) * ogfH))
        B, N, H, W, D, _ = reference_points_cam.shape
        reference_points_cam = reference_points_cam.permute(1, 0, 2, 3, 4, 5).reshape(N, B, H*W, D, 3)
        mask = mask.permute(1, 0, 2, 3, 4, 5).reshape(N, B, H*W, D, 1).squeeze(-1)

        return reference_points, reference_points_cam[..., :2], mask, reference_points_cam[..., 2:3]

    def simple_test(self, **kwargs):
        
        preds_plan_dicts = self.inner_forward(**kwargs)
        pred_traj = self.get_bboxes(
            preds_plan_dicts, **kwargs
        )

        img_metas = kwargs['img_metas']
        output_list = [dict() for _ in range(len(img_metas))]
        for i, result_dict in enumerate(output_list):
            result_dict['pred_ego_traj'] = pred_traj[i]
            result_dict['index'] = img_metas[i]['index']
        
        pred_ego_fut_trajs = output_list[0]['pred_ego_traj']['pred_ego_fut_trajs']


        if not self.training:
            pred_ego_fut_trajs_ = torch.cat([pred_ego_fut_trajs.new_zeros(1, 2), pred_ego_fut_trajs], 0)
            rotate_angle_list=[]
            rotate_angle = 0
            for i in range(pred_ego_fut_trajs_.size(0)-1):
                delta = pred_ego_fut_trajs_[i+1] - pred_ego_fut_trajs_[i]
                cur_rotate_angle = torch.atan2(*delta[[1, 0]])
                if delta.norm()<1: cur_rotate_angle = 0
                rotate_angle = cur_rotate_angle
                rotate_angle_list.append(rotate_angle)
            fut_gt_bboxes_3d = kwargs['fut_boxes_in_cur_ego_list'][0][0]
            rgb_image_list = []
            rgb_image, front_img = self.visual_sample(output_list,  gt_bboxes_3d_=kwargs['gt_bboxes_3d'][0][0], ego_info=None, 
                cam_params=kwargs['img_inputs'][0][1:],
                front_img=kwargs['img_inputs'][0][0][0, 1],
                metric_dict = pred_traj[0]['metric_dict'],
                **kwargs)
            print(f'sc_{img_metas[0]["index"]}')
            # mmcv.imwrite(rgb_image, f'sc_{img_metas[0]["index"]}.png')
            # mmcv.mkdir_or_exist(f'vis/{img_metas[0]["scene_name"]}/')
            mmcv.imwrite(front_img, f'vis/go_stright/{img_metas[0]["scene_name"]}/{img_metas[0]["index"]}.jpg')
            # for i, gt_bboxes_3d in enumerate(fut_gt_bboxes_3d):
                
            #     ego_info = [pred_ego_fut_trajs[i][0].item(), pred_ego_fut_trajs[i][1].item(), 0], [1.85, 4.084, 1], rotate_angle_list[i].item()
            #     rgb_image = self.visual_sample(output_list,  gt_bboxes_3d_=gt_bboxes_3d, ego_info=ego_info, **kwargs)
            #     rgb_image_list.append(rgb_image)
        
        return output_list

    def extract_feat(self): pass
    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas=None,  rescale=False, gt_ego_fut_trajs=None, gt_ego_fut_cmd=None, gt_ego_fut_masks=None,  gt_fut_segmentations_plus=None, gt_fut_segmentations=None, vad_ego_fut_trajs=None, **kwargs):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        pred_ego_fut_trajs = preds_dicts['ego_fut_preds']
        gt_ego_fut_trajs = torch.stack(gt_ego_fut_trajs).to(pred_ego_fut_trajs.device)
        gt_ego_fut_cmd = torch.stack(gt_ego_fut_cmd).to(pred_ego_fut_trajs.device)
        gt_ego_fut_masks = torch.stack(gt_ego_fut_masks).to(pred_ego_fut_trajs.device)

        pred_ego_fut_trajs = torch.cumsum(pred_ego_fut_trajs[gt_ego_fut_cmd==1], 1)
        # pred_ego_fut_trajs = vad_ego_fut_trajs[0][None]
        pred_ego_fut_trajs = preds_dicts['fut_traj_from_velo']
        ego_trajs = torch.cat([torch.zeros_like(pred_ego_fut_trajs[:,:1]), pred_ego_fut_trajs], 1)
        ego_trajs = torch.cat([ego_trajs, torch.zeros_like(ego_trajs[..., :1])], -1)
        ego_trajs_in_global = transform_reference_points(ego_trajs, preds_dicts['data']['ego_pose'], reverse=False)[..., :2]

        # pred_ego_fut_trajs = gt_ego_fut_trajs
        metric_dict = {
            'plan_L2_1s':0,
            'plan_L2_2s':0,
            'plan_L2_3s':0,
            'plan_obj_col_1s':0,
            'plan_obj_col_2s':0,
            'plan_obj_col_3s':0,
            'plan_obj_box_col_1s':0,
            'plan_obj_box_col_2s':0,
            'plan_obj_box_col_3s':0,
            'plan_obj_col_plus_1s':0,
            'plan_obj_col_plus_2s':0,
            'plan_obj_col_plus_3s':0,
            'plan_obj_box_col_plus_1s':0,
            'plan_obj_box_col_plus_2s':0,
            'plan_obj_box_col_plus_3s':0,
            'l2_dist': 0,
        }
        

        fut_valid_flag = gt_ego_fut_masks.all()
        future_second = 3
        metric_dict['fut_valid_flag'] = fut_valid_flag.cpu().item()
        for i in range(future_second):
            if fut_valid_flag:
                cur_time = (i+1)*2
                traj_L2 = self.planning_metric.compute_L2(
                    pred_ego_fut_trajs[0, :cur_time].detach().to(gt_ego_fut_trajs.device),
                    gt_ego_fut_trajs[0, :cur_time]
                )

                obj_coll, obj_box_coll = self.planning_metric.evaluate_coll(
                    pred_ego_fut_trajs[:, :cur_time].detach().to(gt_ego_fut_trajs.device),
                    gt_ego_fut_trajs[:, :cur_time],
                    gt_fut_segmentations,
                    index = [each['index'] for each in img_metas],
                    ignore_gt=False,
                )
                metric_dict['plan_L2_{}s'.format(i+1)] = traj_L2
                metric_dict['plan_obj_col_{}s'.format(i+1)] = obj_coll.mean().item()
                metric_dict['plan_obj_box_col_{}s'.format(i+1)] = obj_box_coll.max().item()
        
        for i in range(future_second):
            if fut_valid_flag:
                cur_time = (i+1)*2
                obj_coll, obj_box_coll = self.planning_metric.evaluate_coll(
                    pred_ego_fut_trajs[:, :cur_time].detach().to(gt_ego_fut_trajs.device),
                    gt_ego_fut_trajs[:, :cur_time],
                    gt_fut_segmentations_plus,
                    index = [each['index'] for each in img_metas],
                    ignore_gt=False,
                )
                metric_dict['plan_obj_col_plus_{}s'.format(i+1)] = obj_coll.mean().item()
                metric_dict['plan_obj_box_col_plus_{}s'.format(i+1)] = obj_box_coll.max().item()

        l2_dist = (pred_ego_fut_trajs-gt_ego_fut_trajs).norm(dim=-1) * gt_ego_fut_masks[:, None]

        l2_dist[gt_ego_fut_masks[:, None]==0] = -1
        metric_dict['l2_dist'] = l2_dist[0].cpu()
        ret_list = []
        num_samples = len(pred_ego_fut_trajs)
        assert num_samples == 1
        
        index_w_scene = img_metas[0]['scene_name'] + '-' + str(img_metas[0]['index'])

        for i in range(num_samples):
            ret_list.append(
                dict(
                    pred_ego_fut_trajs = pred_ego_fut_trajs[i].cpu(),
                    gt_ego_fut_trajs = gt_ego_fut_trajs[i].cpu(),
                    metric_dict = metric_dict,
                    l2_dist=l2_dist[i].cpu(),
                    index_w_scene = index_w_scene,
                    ego_trajs_in_global = ego_trajs_in_global[i].cpu(),
                    gt_ego_fut_cmd = gt_ego_fut_cmd[i].cpu(),
                    index = img_metas[i]['index']
                    )
                )
        return ret_list

    def world2bev_vis(self, x, y):
             return int((x + 50) * 5), int((y + 50) * 5)

    def visual_sample(self, results, gt_bboxes_3d_=None, ego_info=None, cam_params=None, front_img=None,
            metric_dict = None,
            **kwargs):

        import matplotlib.pyplot as plt
        import random
        import math
        import pyquaternion
        from nuscenes.utils.data_classes import Box as NuScenesBox
        from mmdet3d.core.bbox import CustomBox

        # nusc = NuScenes(version='v1.0-trainval', dataroot='./data/nuscenes', verbose=True)
        # _, boxes_gt, _ = nusc.get_sample_data(sample_data_token, box_vis_level=box_vis_level)

        ratio=1
        # plt.figure(figsize=(10, 10*ratio), dpi=300)
        fig, axes = plt.subplots(1, 1, figsize=(10, 10*ratio), dpi=300)
        plt.gca().set_axis_off()
        plt.axis('off')
        fig.tight_layout()

        margin=50.0
        coor_range = self.world2bev_vis(-margin, margin)
        axes.set_xlim(np.array(coor_range))
        axes.set_ylim(np.array(coor_range))
        axes.grid(False)
        # ax = plt.gca()
        axes.set_aspect('equal', adjustable='box')  
        axes.invert_yaxis()
        random.seed(0)
        colors = ['#%02X%02X%02X' % (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255) ) for _ in range(40)]

        ego_center =  self.world2bev_vis(0, 0)
        axes.scatter(ego_center[0], ego_center[1], s=15, marker='o',color='r', zorder=2)

        if gt_bboxes_3d_ is not None:
            # gt_bboxes_3d = kwargs['gt_bboxes_3d'][0][0]
            # bev_coor = gt_bboxes_3d.bev.cpu().numpy()
            # rects = [(tuple(coor[:2]), tuple(coor[2:4]), math.degrees(coor[4])) for coor in bev_coor]
            # boxes = np.array([cv2.boxPoints(rect) for rect in rects])
            # raw = gt_bboxes_3d.corners[:, [4, 7, 3, 0], :2]
            boxes = gt_bboxes_3d_.tensor.numpy().copy()
            for i, box in enumerate(boxes):
                center = box[:3]
                wlh = box[[4, 3, 5]]
                box_yaw = box[6]
                box_vel = box[7:].tolist()
                box_vel.append(0)
                quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw)
                center[:2]=np.array(self.world2bev_vis(center[0],center[1]))
                wlh[0]=wlh[0]*5
                wlh[1]=wlh[1]*5
                nusc_box = CustomBox(center, wlh, quat, velocity=box_vel)
                c = colors[i % len(colors)]
                nusc_box.render(axes, view=np.eye(4), colors=(c, c, c), linewidth=1)
            # if ego_info is not None:
            #     center, wlh, yaw = ego_info
            #     center[:2]=np.array(self.world2bev_vis(center[0],center[1]))
            #     quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=yaw)
            #     wlh[0]=wlh[0]*5
            #     wlh[1]=wlh[1]*5
            #     nusc_box = CustomBox(center, wlh, quat, velocity=[0, 0, 0])
            #     c = colors[-1]
            #     nusc_box.render(axes, view=np.eye(4), colors=(c, c, c), linewidth=1)
        points_per_step=5
        if results[0].get('pred_ego_traj') is not None:
            pred_ego_fut_trajs = results[0]['pred_ego_traj']['pred_ego_fut_trajs']
            # pred_ego_fut_trajs = kwargs['gt_ego_fut_trajs'][0].cpu()
            pred_ego_fut_trajs = pred_ego_fut_trajs.numpy()
            points = np.array([self.world2bev_vis(*point) for point in pred_ego_fut_trajs])          
            points = np.insert(points, 0, np.array(ego_center), axis=0)
           
            points, colors = self._render_traj_v2(points, colormap='autumn')
            x_coords, y_coords = zip(*points)
            for j in range(len(points) - 1):                 
                axes.plot([x_coords[j], x_coords[j + 1]], [y_coords[j], y_coords[j + 1]], '-',c=colors[j], linewidth=1.5, zorder=2)  
                if j != 0 and j % points_per_step==0:
                    axes.scatter(x_coords[j], y_coords[j], s=5, marker='o',color=colors[j], zorder=3)
            axes.scatter(x_coords[-1], y_coords[-1], s=5, marker='o',color=colors[-1], zorder=3)

            if front_img is not None:

                pred_ego_fut_trajs = results[0]['pred_ego_traj']['pred_ego_fut_trajs']
                # pred_ego_fut_trajs = kwargs['gt_ego_fut_trajs'][0].cpu()
                pred_ego_fut_trajs = torch.cat([torch.tensor([[4, 0]]), pred_ego_fut_trajs], 0)
                pred_ego_fut_trajs = torch.cat([pred_ego_fut_trajs, torch.zeros_like(pred_ego_fut_trajs[:, :1])], -1)
                traj_on_img = self.point_sampling(pred_ego_fut_trajs[None, None].to(cam_params[0].device), cam_params)[1][1, 0, 0].cpu().numpy()
                front_img = front_img.permute(1, 2, 0)[:, :, [2, 1, 0]].cpu().numpy()
                front_img = np.ascontiguousarray(front_img, dtype=np.uint8)
               
                traj_on_img, colors = self._render_traj_v2(traj_on_img, colormap='autumn')
                traj_on_img = np.ascontiguousarray(traj_on_img, dtype=np.int32)

                for i in range(len(traj_on_img)-1):
                    front_img = cv2.line(front_img, traj_on_img[i], traj_on_img[i+1] , color=colors[i] * 255, thickness=5)
                
                avg_l2 = 0
                for i in range(1,4):
                    avg_l2 += metric_dict[f'plan_L2_{i}s']
                avg_l2/=3

                avg_coli = 0
                for i in range(1,4):
                    avg_coli += metric_dict[f'plan_obj_box_col_{i}s']
                avg_coli = (avg_coli/3)>0
                
                avg_intersect = 0
                for i in range(1,4):
                    avg_intersect += metric_dict[f'plan_obj_box_col_plus_{i}s']
                avg_intersect = (avg_intersect/3)>0

                # org 
                org = (50, 50) 
                # fontScale 
                fontScale = 1.5
                # Blue color in BGR 
                color = (10, 10, 254) 
                # Line thickness of 2 px 
                thickness = 2
                # Using cv2.putText() method 
                # front_img = cv2.rectangle(front_img, (0, 0), (300, 150), (255, 255, 255), -1)
                front_img = cv2.putText(front_img, 'Avg.L2: %.2f'%avg_l2, (10, 40), cv2.FONT_HERSHEY_SIMPLEX , fontScale, color, thickness, cv2.LINE_AA) 
                # front_img = cv2.putText(front_img, f'Collision: NaN', (10, 90), cv2.FONT_HERSHEY_SIMPLEX , fontScale, color, thickness, cv2.LINE_AA) 
                # front_img = cv2.putText(front_img, f'Intersection: NaN', (10, 140), cv2.FONT_HERSHEY_SIMPLEX , fontScale, color, thickness, cv2.LINE_AA) 
                front_img = cv2.putText(front_img, f'Collision: {str(avg_coli)}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX , fontScale, color, thickness, cv2.LINE_AA) 
                front_img = cv2.putText(front_img, f'Intersection: {str(avg_intersect)}', (10, 140), cv2.FONT_HERSHEY_SIMPLEX , fontScale, color, thickness, cv2.LINE_AA) 
                # mmcv.imwrite(front_img, '')

        if kwargs.get('map_gt_bboxes_3d', False):
            map_gt_bboxes_3d = kwargs['map_gt_bboxes_3d'][0][0]
            map_gt_labels_3d = kwargs['map_gt_labels_3d'][0][0]
            for i, instance in enumerate(map_gt_bboxes_3d.instance_list):
                # if map_gt_labels_3d[i]!=2: continue
                line = np.array(list(instance.coords))
                corners = np.array([self.world2bev_vis(*corner) for corner in line])
                corners = [each for each in corners if ((each>=0).all() & (each<512).all())]
                if len(corners)<1: continue
                x_coords, y_coords = zip(*corners)
                for k, corner in enumerate(corners[:-1]):
                    axes.plot([x_coords[k], x_coords[k + 1]], [y_coords[k], y_coords[k + 1]], c='dimgray', linewidth=1, zorder=1,) 
        
        if kwargs.get('gt_agent_fut_traj', False):
            gt_agent_fut_traj = kwargs['gt_agent_fut_traj'][0][0].cpu()
            gt_agent_fut_traj_mask = kwargs['gt_agent_fut_traj_mask'][0][0].cpu()
            centers = kwargs['gt_bboxes_3d'][0][0].center[..., :2].cpu()
            tmp = torch.cat([centers[:, None], gt_agent_fut_traj], 1)
            trajs = torch.cumsum(tmp, 1)
            for k, traj in enumerate(trajs):               
                traj = traj.cpu().numpy()
                # center = np.array(self.world2bev_vis(*centers[k]))
                agent_fut_traj = np.array([self.world2bev_vis(*corner) for corner in traj])
                corners, colors = self._render_traj_v2(agent_fut_traj, colormap='winter',points_per_step=points_per_step)
                corners = [each for each in corners if ((each>=0).all() & (each<1536).all())]
                x_coords, y_coords = zip(*corners)
                for j in range(len(corners) - 1):
                    # plot line between box center and the first traj point
                    if j//points_per_step == 0 and gt_agent_fut_traj_mask[k, j//points_per_step].sum()==2:
                        axes.plot([x_coords[j], x_coords[j + 1]], [y_coords[j], y_coords[j + 1]], '-',c=colors[j], linewidth=0.8, zorder=2)  
                        continue  
                    elif gt_agent_fut_traj_mask[k, j//points_per_step].sum()<2 or gt_agent_fut_traj_mask[k, j//points_per_step-1].sum()<2:
                        continue                                 
                    axes.plot([x_coords[j], x_coords[j + 1]], [y_coords[j], y_coords[j + 1]], '-',c=colors[j], linewidth=0.8, zorder=2)  


        plt.margins(0, 0)
        # plt.savefig(f'pred_bev_{results[0]["index"]}.png')
        canvas = FigureCanvasAgg(plt.gcf())
        canvas.draw()
        w, h = canvas.get_width_height()
        buf = np.fromstring(canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w, h, 4)
        buf = np.roll(buf, 3, axis=2)
        image = Image.frombytes("RGBA", (w, h), buf.tostring())
        image = np.asarray(image)
        rgb_image = image[:, :, :3]
        plt.close()
        return rgb_image, front_img

    def _render_traj(self, future_traj, traj_score=1, colormap='winter', points_per_step=5, line_color=None, dot_color=None, dot_size=25):
        total_steps = (len(future_traj)-1) * points_per_step + 1
        dot_colors = matplotlib.colormaps[colormap](
            np.linspace(0, 1, total_steps))[:, :3] * 255
        dot_colors = dot_colors*traj_score + \
            (1-traj_score)*np.ones_like(dot_colors)
        total_xy = np.zeros((total_steps, 2))
        for i in range(total_steps-1):
            unit_vec = future_traj[i//points_per_step +
                                   1] - future_traj[i//points_per_step]
            total_xy[i] = (i/points_per_step - i//points_per_step) * \
                unit_vec + future_traj[i//points_per_step]
        total_xy[-1] = future_traj[-1]
        return total_xy, dot_colors

    def _render_traj_v2(self, future_traj, traj_score=1, colormap='winter', points_per_step=5, line_color=None, dot_color=None, dot_size=25):
            total_steps = (len(future_traj)-1) * points_per_step + 1
            dot_colors = matplotlib.colormaps[colormap](
                np.linspace(0, 1, total_steps))[:, :3]
    #         dot_colors = dot_colors*traj_score + \
    #             (1-traj_score)*np.ones_like(dot_colors)
            total_xy = np.zeros((total_steps, 2))
            for i in range(total_steps-1):
                unit_vec = future_traj[i//points_per_step +
                                    1] - future_traj[i//points_per_step]
                total_xy[i] = (i/points_per_step - i//points_per_step) * \
                    unit_vec + future_traj[i//points_per_step]
            total_xy[-1] = future_traj[-1]
            return total_xy, dot_colors