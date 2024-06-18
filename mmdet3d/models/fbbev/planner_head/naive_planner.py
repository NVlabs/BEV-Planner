# Copyright (c) 2023-2024, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# TODO: add license here


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
class NaivePlannerHead(BaseModule):
  
    _version = 2

    def __init__(self,
                 # num_classes=1,
                 in_channels=256,
                 stride=[16],
                 embed_dims=256,
                 num_query=1,
                 num_reg_fcs=2,
                 memory_len=12,
                 topk_proposals=4,
                 num_propagated=0,
                 with_dn=True,
                 with_ego_pos=True,
                 match_with_velo=True,
                 match_costs=None,
                 transformer=None,
                 sync_cls_avg_factor=False,
                 code_weights=None,
                 bbox_coder=None,
                 init_cfg=None,
                 normedlinear=False,
                 point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 loss_plan_reg=dict(type='L1Loss', loss_weight=5.0),
                 loss_plan_col=dict(type='PlanCollisionLoss', loss_weight=5.0),
                 with_ego_status=False,
                 dist_func_type='MDE',
                 use_map_info=False,
                **kwargs):

        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 2
        self.use_map_info = use_map_info

        self.with_ego_status = with_ego_status
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


        super(NaivePlannerHead, self).__init__()
       
        self.pc_range = nn.Parameter(torch.tensor(
            point_cloud_range), requires_grad=False)

        self.loss_plan_reg = build_loss(loss_plan_reg)
        loss_plan_col.update(point_cloud_range=point_cloud_range)
        self.loss_plan_col = build_loss(loss_plan_col)


        ego_img_decoder = dict(
                    type='CustomTransformerDecoder',
                    num_layers=1,
                    return_intermediate=False,
                    transformerlayers=dict(
                        type='BaseTransformerLayer',
                        batch_first=True,
                        attn_cfgs=dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            attn_drop=0.1,
                            proj_drop=0.1,
                        ),
                        feedforward_channels=1024,
                        ffn_dropout=0.1,
                        operation_order=('cross_attn', 'norm', 'ffn', 'norm')))
        if self.use_map_info:
            ego_agent_decoder = dict(
                    type='CustomTransformerDecoder',
                    num_layers=1,
                    return_intermediate=False,
                    transformerlayers=dict(
                        type='BaseTransformerLayer',
                        batch_first=True,
                        attn_cfgs=dict(
                            type='MotionSelfAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1,
                            dist_func_type=dist_func_type,
                            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                            consider_map_quality=True,
                        ),
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    operation_order=('cross_attn', 'norm', 'ffn', 'norm')))

            self.ego_agent_decoder = build_transformer_layer_sequence(ego_agent_decoder)
            self.gamma = nn.Parameter(torch.ones(256)*0.5, requires_grad=True)
        self.ego_img_decoder = build_transformer_layer_sequence(ego_img_decoder)
        # self.ego_decoder = build_transformer_layer_sequence(ego_agent_decoder)

        self.ego_info = MLN(3)
        self._init_layers()
        self.reset_memory()
        self.planning_metric = PlanningMetric()

        self.count = 0
    
    def reset_memory(self):
        self.memory_traj = None
        # self.memory_ego_embed = None

    def pre_update_memory(self, data, fut_traj_from_velo):

        x = 1-data['start_of_sequence'] # original prev_exist, so we need do `not`
        B = x.size(0)
        # refresh the memory when the scene changes
        if self.memory_traj is None:
            self.memory_traj =  fut_traj_from_velo.unsqueeze(1).repeat(1, self.memory_len, 1, 1) * 0
            # self.memory_ego_embed = x.new_zeros(B, self.memory_len, self.embed_dims * 2)
        else:
            self.memory_traj = transform_reference_points(self.memory_traj, data['ego_pose_inv'], reverse=False)[..., :2]
            self.memory_traj = memory_refresh(self.memory_traj[:, :self.memory_len], x) 
            for i in range(B):
                if not x[i]: self.memory_traj[i, 0] = fut_traj_from_velo[i] * 0
            
            # self.memory_ego_embed = memory_refresh(self.memory_ego_embed[:, :self.memory_len], x)

    def post_update_memory(self, data, ego_fut_trajs, ego_embeds):
        self.memory_traj = torch.cat([ego_fut_trajs, self.memory_traj], dim=1)
        self.memory_traj = torch.cat([self.memory_traj, torch.zeros_like(self.memory_traj[..., :1])], -1)
        self.memory_traj = transform_reference_points(self.memory_traj, data['ego_pose'], reverse=False)
        # self.memory_ego_embed = torch.cat([ego_embeds, self.memory_ego_embed], dim=1)
    
    def _init_layers(self):
        """Initialize layers of the transformer head."""

        ego_fut_decoder = []
        ego_fut_dec_in_dim = self.embed_dims
        if self.with_ego_status:
            ego_fut_dec_in_dim += 9
        
        for i in range(self.num_reg_fcs):
            if i ==0: 
                ego_fut_decoder.append(Linear(ego_fut_dec_in_dim, self.embed_dims))
            else:
                ego_fut_decoder.append(Linear(self.embed_dims, self.embed_dims))
            ego_fut_decoder.append(nn.ReLU())
        ego_fut_decoder.append(Linear(self.embed_dims, self.ego_fut_mode*self.fut_steps*2))
        self.ego_fut_decoder = nn.Sequential(*ego_fut_decoder)

        self.query_feat_embedding = nn.Embedding(self.num_query, self.embed_dims)

    def calc_MDE(self, reference_points_q, reference_points_v, pc_range, map_scores=None):
        """
        mim mean distance between the map lane and traj.
        """

        reference_points_q = reference_points_q[..., :2]
        q_shape = reference_points_q.shape
        v_shape = reference_points_v.shape
        reference_points_q = reference_points_q.flatten(1, 2)
        reference_points_v = reference_points_v.flatten(1, 2)
        
        dist = []
        code_size = reference_points_q.size(-1)
        for b in range(reference_points_q.shape[0]):
            dist_b = torch.norm(reference_points_q[b].reshape(-1, 1, code_size) - reference_points_v[b].reshape(1, -1, code_size), dim=-1)
            dist.append(dist_b[None, ...])
        dist = torch.cat(dist, dim=0)  # [B, Q, K]
        dist = dist.view(q_shape[0], q_shape[1], q_shape[2], v_shape[1], v_shape[2])
        dist = dist.min(-1).values.mean(2)
        
        if map_scores is not None:
            map_scores = map_scores.sigmoid().max(-1)[0] # smaller, better
            map_scores = torch.round(1-map_scores, decimals=1) + self.map_alpha
            dist = dist * map_scores.unsqueeze(1)
            
        dist = -dist

        return dist

    def forward(self, results, gt_ego_lcf_feat, gt_ego_fut_cmd, gt_ego_his_traj=None, gt_ego_fut_trajs=None, img_metas=None, map_results=None):
        
        # agent_queries = map_results['queries']
        if self.use_map_info:
            map_queries = map_results['queries'].clone()
            map_lines = map_results['lines'].clone()
            map_scores = map_results['scores'].clone()
            B, NMQ, K2 = map_lines.shape
            map_lines = map_lines.reshape(B, NMQ, K2//2, 2)
            # map_pos = self.query_embedding(bevpos2posemb(map_lines.mean(-2)))
            map_lines = get_ego_pos(map_lines, self.pc_range)

        img_context = results['img_bev_feat'][0].flatten(-2, -1).permute(0, 2, 1)
        
        gt_ego_lcf_feat = torch.stack(gt_ego_lcf_feat).to(img_context.device)
        gt_ego_fut_cmd = torch.stack(gt_ego_fut_cmd).to(img_context.device)

        start_of_sequence = torch.FloatTensor([
            single_img_metas['start_of_sequence'] 
            for single_img_metas in img_metas]).to(img_context.device)

        timestamp = torch.FloatTensor([
            single_img_metas['timestamp'] 
            for single_img_metas in img_metas]).to(img_context.device)

        ego_pose_inv = torch.stack([
            single_img_metas['ego_pose_inv'] 
            for single_img_metas in img_metas], 0).to(img_context.device)

        ego_pose = torch.stack([
            single_img_metas['ego_pose'] 
            for single_img_metas in img_metas], 0).to(img_context.device)

        data = dict(
            start_of_sequence = start_of_sequence,
            timestamp = timestamp,
            ego_pose_inv = ego_pose_inv,
            ego_pose = ego_pose,
        )

        fut_traj_from_velo = gt_ego_lcf_feat[:, :2].unsqueeze(1).repeat(1, self.fut_steps, 1) * torch.arange(1, self.fut_steps+1)[None,:, None].to(img_context.device) * 0.5

        self.pre_update_memory(data, fut_traj_from_velo)
        bs = img_context.size(0)
        ego_query = self.query_feat_embedding.weight.repeat(bs, 1)
        ego_query = self.ego_info(ego_query, gt_ego_fut_cmd.to(ego_query.dtype)).unsqueeze(1)

        init_ego_traj =  self.memory_traj[:, 0:1]

        if self.use_map_info:
            ego_query = (1-self.gamma) * self.ego_agent_decoder(
                query = ego_query,
                key = map_queries,
                val = map_queries,
                reference_points_q=init_ego_traj,
                reference_points_v=map_lines,
                pc_range=self.pc_range,
                map_scores=map_scores
                )   + self.gamma * self.ego_img_decoder(
                query = ego_query,
                key = img_context,
                val = img_context,
                # query_pos = ego_pose
                )
        else:
            ego_query =self.ego_img_decoder(
                query = ego_query,
                key = img_context,
                val = img_context,
                )
        if self.with_ego_status:
            ego_query = torch.cat([ego_query[:, 0], gt_ego_lcf_feat], -1)
        outputs_ego_trajs = self.ego_fut_decoder(ego_query)
        outputs_ego_trajs = outputs_ego_trajs.reshape(outputs_ego_trajs.shape[0], 
                                                      self.ego_fut_mode, self.fut_steps, 2)


        self.post_update_memory(data, torch.cumsum(outputs_ego_trajs[gt_ego_fut_cmd==1], 1)[:, None], ego_query)
        

        ego_trajs = torch.cumsum(outputs_ego_trajs[gt_ego_fut_cmd==1], 1)
        ego_trajs = torch.cat([torch.zeros_like(ego_trajs[:,:1]), ego_trajs], 1)
        ego_trajs = torch.cat([ego_trajs, torch.zeros_like(ego_trajs[..., :1])], -1)
        ego_trajs_in_global = transform_reference_points(ego_trajs, data['ego_pose'], reverse=False)[..., :2]

        return dict(
            ego_fut_preds=outputs_ego_trajs,
            ego_trajs_in_global = ego_trajs_in_global,
            data=data
        )

    @force_fp32(apply_to=('preds_plan_dicts'))
    def loss(self,
             gt_ego_fut_trajs=None,
             gt_ego_fut_cmd=None,
             gt_ego_fut_masks=None,
             preds_plan_dicts=None,
             img_metas=None,
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


    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas,  rescale=False, gt_ego_fut_trajs=None, 
        gt_ego_fut_cmd=None, gt_ego_fut_masks=None, gt_fut_segmentations=None, gt_fut_segmentations_plus=None,
        vad_ego_fut_trajs=None, **kwargs,
        ):
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
        ego_trajs = torch.cat([torch.zeros_like(pred_ego_fut_trajs[:,:1]), pred_ego_fut_trajs], 1)
        ego_trajs = torch.cat([ego_trajs, torch.zeros_like(ego_trajs[..., :1])], -1)
        ego_trajs_in_global = transform_reference_points(ego_trajs, preds_dicts['data']['ego_pose'], reverse=False)[..., :2]
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
                metric_dict['plan_obj_col_{}s'.format(i+1)] = obj_coll.max().item()
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


class MLN(nn.Module):
    ''' 
    Args:
        c_dim (int): dimension of latent code c
        f_dim (int): feature dimension
    '''

    def __init__(self, c_dim, f_dim=256, use_ln=True):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.use_ln = use_ln

        self.reduce = nn.Sequential(
            nn.Linear(c_dim, f_dim),
            nn.ReLU(),
        )
        self.gamma = nn.Linear(f_dim, f_dim)
        self.beta = nn.Linear(f_dim, f_dim)
        if self.use_ln:
            self.ln = nn.LayerNorm(f_dim, elementwise_affine=False)
        self.init_weight()

    def init_weight(self):
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, c):
        if self.use_ln:
            x = self.ln(x)
        c = self.reduce(c)
        gamma = self.gamma(c)
        beta = self.beta(c)
        out = gamma * x + beta

        return out