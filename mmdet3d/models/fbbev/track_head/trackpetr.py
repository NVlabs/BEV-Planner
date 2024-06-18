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
from .streampetr_utils import *
from .instances import Instances
from .runtime_tracker import RunTimeTracker
import copy
from mmdet.models.utils import NormedLinear
from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from mmdet3d.models.fbbev.utils import save_tensor

@HEADS.register_module()
class TackerHead(AnchorFreeHead):
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
                 num_classes,
                 in_channels=256,
                 stride=[16],
                 embed_dims=256,
                 num_query=100,
                 num_reg_fcs=2,
                 memory_len=1024,
                 topk_proposals=256,
                 num_propagated=256,
                 with_dn=True,
                 with_ego_pos=True,
                 match_with_velo=True,
                 match_costs=None,
                 transformer=None,
                 sync_cls_avg_factor=False,
                 code_weights=None,
                 bbox_coder=None,
                loss=dict(
                    type='TrackingLossCombo',
                    num_classes=10,
                    interm_loss=True,
                    code_weights = [2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    loss_cls=dict(
                        type='FocalLoss',
                        use_sigmoid=True,
                        gamma=2.0,
                        alpha=0.25,
                        loss_weight=2.0),
                    loss_bbox=dict(type='L1Loss', loss_weight=0.25),
                    loss_iou=dict(type='GIoULoss', loss_weight=0.0),
                    # loss_prediction=dict(type='L1Loss', loss_weight=0.5),
                    assigner=dict(
                        type='HungarianAssigner3D',
                        cls_cost=dict(type='FocalLossCost', weight=2.0),
                        reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                        iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head. 
                    pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
                    ),
                    train_cfg=dict(
                     assigner=dict(
                         type='HungarianAssigner3D',
                         cls_cost=dict(type='ClassificationCost', weight=1.),
                         reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                         iou_cost=dict(
                             type='IoUCost', iou_mode='giou', weight=2.0)),),
                 test_cfg=dict(max_per_img=100),
                 scalar = 5,
                 noise_scale = 0.4,
                 noise_trans = 0.0,
                 dn_weight = 1.0,
                 split = 0.5,
                 init_cfg=None,
                 normedlinear=False,
                 runtime_tracker=dict(
                        output_threshold=0.2,
                        score_threshold=0.2,
                        record_threshold=0.4,
                        max_age_since_update=7),
                 tracking=True,
                 layer_index=-1,
                 **kwargs):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
            
        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.memory_len = memory_len
        self.topk_proposals = topk_proposals
        self.num_propagated = num_propagated
        self.with_dn = with_dn
        self.with_ego_pos = with_ego_pos
        self.match_with_velo = match_with_velo
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.embed_dims = embed_dims
        self.with_dn = with_dn
        self.stride=stride
        self.layer_index = layer_index
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10

        self.scalar = scalar
        self.bbox_noise_scale = noise_scale
        self.bbox_noise_trans = noise_trans
        self.dn_weight = dn_weight
        self.split = split 

        self.act_cfg = transformer.get('act_cfg',
                                       dict(type='ReLU', inplace=True))
        self.num_pred = transformer['decoder']['num_layers']
        self.normedlinear = normedlinear
        self.tracking = tracking 
        super(TackerHead, self).__init__(num_classes, in_channels, init_cfg = init_cfg)

        self.criterion = build_loss(loss)

        self.transformer = build_transformer(transformer)

        self.bbox_coder = build_bbox_coder(bbox_coder)

        self.pc_range = nn.Parameter(torch.tensor(
            self.bbox_coder.pc_range), requires_grad=False)

        self._init_layers()
        self.reset_history_track_instances()
        
        self.count = 0
        
        self.hist_len = 4
        # self.fut_len = 8
        if runtime_tracker:
            self.runtime_tracker = RunTimeTracker(**runtime_tracker)
            self.runtime_tracker.empty()


    def _init_layers(self):
        """Initialize layers of the transformer head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        if self.normedlinear:
            cls_branch.append(NormedLinear(self.embed_dims, self.cls_out_channels))
        else:
            cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        # self.cls_branches = nn.ModuleList(
        #     [fc_cls for _ in range(self.num_pred)])
        # self.reg_branches = nn.ModuleList(
        #     [reg_branch for _ in range(self.num_pred)])

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


        self.cls_branches =_get_clones(fc_cls, self.num_pred)
        self.reg_branches = _get_clones(reg_branch, self.num_pred)
        self.reference_points = nn.Embedding(self.num_query, 3)
        if self.num_propagated > 0:
            self.pseudo_reference_points = nn.Embedding(self.num_propagated, 3)


        self.query_embedding = nn.Sequential(
            nn.Linear(self.embed_dims*3//2, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        if self.tracking:
            self.query_feat_embedding = nn.Embedding(self.num_query, self.embed_dims)
        # self.spatial_alignment = MLN(14, use_ln=False)

        self.time_embedding = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims)
        )

        # encoding ego pose
        if self.with_ego_pos:
            self.ego_pose_pe = MLN(180)
            self.ego_pose_memory = MLN(180)

    def init_weights(self):
        """Initialize weights of the transformer head."""
        # The initialization for transformer is important
        nn.init.uniform_(self.reference_points.weight.data, 0, 1)
        if self.num_propagated > 0:
            nn.init.uniform_(self.pseudo_reference_points.weight.data, 0, 1)
            self.pseudo_reference_points.weight.requires_grad = False
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)


    def reset_history_track_instances(self):
        self.history_track_instances = None

    def generate_empty_instance(self, B, init_memory_instances=False):
        """Generate empty instance slots at the beginning of tracking"""
        track_instances = Instances((1, 1))
        device = self.reference_points.weight.device
        """Detection queries"""
        # reference points, query embeds, and query targets (features)
        if init_memory_instances:
            reference_points = self.reference_points.weight.new_zeros(self.memory_len, 3)[None].repeat(B, 1, 1)
            len_track_instances = self.memory_len
        else:
            reference_points = self.reference_points.weight[None].repeat(B, 1, 1)
            len_track_instances = self.num_query
        query_pos = self.query_embedding(pos2posemb3d(reference_points))
        track_instances.reference_points = reference_points.clone()
        track_instances.query_pos = query_pos.clone()
        if self.tracking:
            if init_memory_instances:
                track_instances.query_feats = self.query_feat_embedding.weight.new_zeros(len_track_instances, self.embed_dims)[None].repeat(B, 1, 1)
            else:
                track_instances.query_feats = self.query_feat_embedding.weight.clone()[None].repeat(B, 1, 1)     
        else:
            track_instances.query_feats = torch.zeros_like(query_pos)

        """ StreamPETR memory information"""
        track_instances.timestamp = torch.zeros(B, len_track_instances, 1, dtype=torch.float, device=device)
        track_instances.ego_pose = torch.zeros(B, len_track_instances, 4, 4, dtype=torch.float, device=device)
        track_instances.velo = torch.zeros(B, len_track_instances, 2, dtype=torch.float, device=device)

        """Tracking information"""
        # id for the tracks
        track_instances.obj_idxes = torch.full(
            (B, len_track_instances,), -1, dtype=torch.long, device=device)
        # matched gt indexes, for loss computation
        track_instances.matched_gt_idxes = torch.full(
            (B, len_track_instances,), -1, dtype=torch.long, device=device)
        # life cycle management
        track_instances.disappear_time = torch.zeros(
            (B, len_track_instances, ), dtype=torch.long, device=device)
        track_instances.track_query_mask = torch.zeros(
            (B, len_track_instances, ), dtype=torch.bool, device=device)
        
        """Current frame information"""
        # classification scores
        track_instances.logits = torch.zeros(
            (B, len_track_instances, self.num_classes), dtype=torch.float, device=device)
        # bounding boxes
        track_instances.bboxes = torch.zeros(
            (B, len_track_instances, 10), dtype=torch.float, device=device)
        # track scores, normally the scores for the highest class
        track_instances.scores = torch.zeros(
            (B, len_track_instances, 1), dtype=torch.float, device=device)
        
        # # motion prediction, not normalized
        # track_instances.motion_predictions = torch.zeros(
        #     (B, len_track_instances, self.fut_len, 2), dtype=torch.float, device=device)
        # """Cache for current frame information, loading temporary data for spatial-temporal reasoining"""
        # track_instances.cache_logits = torch.zeros(
        #     (B, len_track_instances, self.num_classes), dtype=torch.float, device=device)
        # track_instances.cache_bboxes = torch.zeros(
        #     (B, len_track_instances, 10), dtype=torch.float, device=device)
        # track_instances.cache_scores = torch.zeros(
        #     (B, len_track_instances,), dtype=torch.float, device=device)
        # track_instances.cache_reference_points = reference_points.clone()
        # track_instances.cache_query_pos = query_pos.clone()
        # if self.tracking:
        #     track_instances.cache_query_feats = self.query_feat_embedding.weight.clone()[None].repeat(B, 1, 1)
        # else:
        #     track_instances.cache_query_feats = torch.zeros_like(query_pos)
        # track_instances.cache_motion_predictions = torch.zeros_like(track_instances.motion_predictions)
        # """History Reasoning"""
        # # embeddings
        track_instances.hist_query_feats = torch.zeros(
            (B, len_track_instances, self.hist_len, self.embed_dims), dtype=torch.float32, device=device)
        # # padding mask, follow MultiHeadAttention, 1 indicates padded
        # track_instances.hist_padding_masks = torch.ones(
        #     (B, len_track_instances, self.hist_len), dtype=torch.bool, device=device)
        # # positions, global coord
        track_instances.hist_xyz = torch.zeros(
            (B, len_track_instances, self.hist_len, 3), dtype=torch.float, device=device)
        # # positional embeds
        # track_instances.hist_position_embeds = torch.zeros(
        #     (B, len_track_instances, self.hist_len, self.embed_dims), dtype=torch.float32, device=device)
        # # bboxes
        track_instances.hist_velo = torch.zeros(
           (B, len_track_instances, self.hist_len, 2), dtype=torch.float, device=device)
        

        track_instances.hist_mask = torch.zeros(
           (B, len_track_instances, self.hist_len), dtype=torch.float, device=device)

        # # logits
        # track_instances.hist_logits = torch.zeros(
        #     (B, len_track_instances, self.hist_len, self.num_classes), dtype=torch.float, device=device)
        # # scores
        # track_instances.hist_scores = torch.zeros(
        #     (B, len_track_instances, self.hist_len), dtype=torch.float, device=device)

        # """Future Reasoning"""
        # # embeddings
        # track_instances.fut_embeds = torch.zeros(
        #     (B, len_track_instances, self.fut_len, self.embed_dims), dtype=torch.float32, device=device)
        # # padding mask, follow MultiHeadAttention, 1 indicates padded
        # track_instances.fut_padding_masks = torch.ones(
        #     (B, len_track_instances, self.fut_len), dtype=torch.bool, device=device)
        # # positions
        # track_instances.fut_xyz = torch.zeros(
        #     (B, len_track_instances, self.fut_len, 3), dtype=torch.float, device=device)
        # # positional embeds
        # track_instances.fut_position_embeds = torch.zeros(
        #     (B, len_track_instances, self.fut_len, self.embed_dims), dtype=torch.float32, device=device)
        # # bboxes
        # track_instances.fut_bboxes = torch.zeros(
        #     (B, len_track_instances, self.fut_len, 10), dtype=torch.float, device=device)
        # # logits
        # track_instances.fut_logits = torch.zeros(
        #     (B, len_track_instances, self.fut_len, self.num_classes), dtype=torch.float, device=device)
        # # scores
        # track_instances.fut_scores = torch.zeros(
        #     (B, len_track_instances, self.fut_len), dtype=torch.float, device=device)
        return track_instances

    def instance_temporal_alignment(self):
        B = self.track_instances.query_pos.size(0)
        temp_history_track_instances = self.history_track_instances.clone()
        temp_reference_points = (temp_history_track_instances.reference_points - self.pc_range[:3]) / (self.pc_range[3:6] - self.pc_range[0:3])

        temp_history_track_instances.query_pos = self.query_embedding(pos2posemb3d(temp_reference_points)) 
        rec_ego_pose = torch.eye(4, device= self.track_instances.query_pos.device).unsqueeze(0).unsqueeze(0).repeat(B,  self.track_instances.query_pos.size(1), 1, 1)
        tmp_ego_pose = torch.eye(4, device= self.track_instances.query_pos.device).unsqueeze(0).unsqueeze(0).repeat(B,  temp_history_track_instances.query_pos.size(1), 1, 1)
        if self.with_ego_pos:
            "current ego pose"
            rec_ego_motion = torch.cat([torch.zeros_like(self.track_instances.reference_points[...,:3]), rec_ego_pose[..., :3, :].flatten(-2)], dim=-1)
            rec_ego_motion = nerf_positional_encoding(rec_ego_motion)
            self.track_instances.query_pos = self.ego_pose_pe(self.track_instances.query_pos, rec_ego_motion)
            self.track_instances.query_feats = self.ego_pose_memory(self.track_instances.query_feats, rec_ego_motion)
            
            "memory ego pose"
            memory_ego_motion = torch.cat([ temp_history_track_instances.velo, temp_history_track_instances.timestamp,  tmp_ego_pose[..., :3, :].flatten(-2)], dim=-1).float()
            memory_ego_motion = nerf_positional_encoding(memory_ego_motion)

            temp_history_track_instances.query_pos = self.ego_pose_pe(temp_history_track_instances.query_pos, memory_ego_motion)
            temp_history_track_instances.query_feats = self.ego_pose_memory(temp_history_track_instances.query_feats, memory_ego_motion)

        self.track_instances.query_pos += self.time_embedding(pos2posemb1d(torch.zeros_like(self.track_instances.reference_points[...,:1])))
        temp_history_track_instances.query_pos += self.time_embedding(pos2posemb1d(temp_history_track_instances.timestamp).float())
        
        if self.num_propagated > 0:
            reference_points = torch.cat([self.track_instances.reference_points, temp_reference_points[:, :self.num_propagated]], dim=1)
            self.track_instances = Instances.cat([self.track_instances, temp_history_track_instances[:, :self.num_propagated]], dim=1)
            temp_history_track_instances = temp_history_track_instances[:, self.num_propagated:]
            temp_reference_points = temp_reference_points[:, self.num_propagated:]
            rec_ego_pose = torch.eye(4, device=self.track_instances.query_pos.device).unsqueeze(0).unsqueeze(0).repeat(B, self.track_instances.query_pos.shape[1], 1, 1)
            
        return reference_points, temp_history_track_instances, temp_reference_points, rec_ego_pose

    def pre_update_instances(self, data):
        x = 1-data['start_of_sequence'] # original prev_exist, so we need do `not`
        B = x.size(0)
        self.track_instances = self.generate_empty_instance(B, init_memory_instances=False)
        if self.history_track_instances is None:
            self.history_track_instances = self.generate_empty_instance(B, init_memory_instances=True)
        else:
            self.history_track_instances.timestamp += data['timestamp'].unsqueeze(-1).unsqueeze(-1)
            self.history_track_instances.ego_pose = data['ego_pose_inv'].unsqueeze(1) @ self.history_track_instances.ego_pose
            self.history_track_instances.reference_points = transform_reference_points(self.history_track_instances.reference_points, data['ego_pose_inv'], reverse=False)
            
            ## hist
            self.history_track_instances.hist_xyz = transform_reference_points(self.history_track_instances.hist_xyz, data['ego_pose_inv'], reverse=False)
            self.history_track_instances.hist_velo = transform_velo(self.history_track_instances.hist_velo,  data['ego_pose_inv'], reverse=False)
            # hist

            self.history_track_instances = self.history_track_instances[:, :self.memory_len]
            
            if data['start_of_sequence'].any():
                self.history_track_instances.timestamp = memory_refresh(self.history_track_instances.timestamp, x)
                self.history_track_instances.reference_points = memory_refresh(self.history_track_instances.reference_points, x)
                self.history_track_instances.query_feats = memory_refresh(self.history_track_instances.query_feats, x)
                self.history_track_instances.ego_pose = memory_refresh(self.history_track_instances.ego_pose, x)
                self.history_track_instances.velo = memory_refresh(self.history_track_instances.velo, x)
                self.history_track_instances.scores = memory_refresh(self.history_track_instances.scores, x)

                ## hist
                self.history_track_instances.hist_xyz = memory_refresh(self.history_track_instances.hist_xyz, x)
                self.history_track_instances.hist_velo = memory_refresh(self.history_track_instances.hist_velo, x)
                self.history_track_instances.hist_mask = memory_refresh(self.history_track_instances.hist_mask, x)
                ##
                device = self.reference_points.weight.device
              
                self.history_track_instances.matched_gt_idxes = (memory_refresh(self.history_track_instances.matched_gt_idxes, x) + (1 - x).view(B, 1) *  torch.full(
                    (B, self.memory_len,), -1, dtype=torch.long, device=device)).to(torch.long)
                self.history_track_instances.obj_idxes = (memory_refresh(self.history_track_instances.obj_idxes, x) + (1 - x).view(B, 1) *  torch.full(
                    (B, self.memory_len,), -1, dtype=torch.long, device=device)).to(torch.long)
        # for the first frame, padding pseudo_reference_points (non-learnable)
        if self.num_propagated > 0:
            pseudo_reference_points = self.pseudo_reference_points.weight * (self.pc_range[3:6] - self.pc_range[0:3]) + self.pc_range[0:3]
            self.history_track_instances.reference_points[:, :self.num_propagated] = self.history_track_instances.reference_points[:, :self.num_propagated] + (1 - x).view(B, 1, 1) * pseudo_reference_points
            self.history_track_instances.ego_pose[:, :self.num_propagated] = self.history_track_instances.ego_pose[:, :self.num_propagated] + (1 - x).view(B, 1, 1, 1) * torch.eye(4, device=x.device)

    def post_update_instances(self, data, rec_ego_pose, all_cls_scores, all_bbox_preds, outs_dec, mask_dict):

        if self.training and mask_dict and mask_dict['pad_size'] > 0:
            rec_reference_points = all_bbox_preds[:, :, mask_dict['pad_size']:, :3][self.layer_index]
            rec_velo = all_bbox_preds[:, :, mask_dict['pad_size']:, -2:][self.layer_index]
            rec_memory = outs_dec[:, :, mask_dict['pad_size']:, :][self.layer_index]
            rec_score = all_cls_scores[:, :, mask_dict['pad_size']:, :][self.layer_index].sigmoid().topk(1, dim=-1).values[..., 0:1]
            rec_timestamp = torch.zeros_like(rec_score, dtype=torch.float64)
            rec_bboxes = all_bbox_preds[:, :, mask_dict['pad_size']:, :][self.layer_index]
        else:
            rec_reference_points = all_bbox_preds[..., :3][self.layer_index]
            rec_velo = all_bbox_preds[..., -2:][self.layer_index]
            rec_memory = outs_dec[self.layer_index]
            rec_score = all_cls_scores[self.layer_index].sigmoid().topk(1, dim=-1).values[..., 0:1]
            rec_timestamp = torch.zeros_like(rec_score, dtype=torch.float64)
            rec_bboxes = all_bbox_preds[self.layer_index]
        
        # topk proposals
        self.track_instances.timestamp = rec_timestamp
        self.track_instances.query_feats = rec_memory
        self.track_instances.ego_pose = rec_ego_pose
        self.track_instances.velo = rec_velo 
        self.track_instances.reference_points = rec_reference_points
        self.track_instances.scores = rec_score
        self.track_instances.bboxes = rec_bboxes

        ## update hist
        self.track_instances.hist_xyz =  torch.cat([self.track_instances.hist_xyz[:, :, 1:], rec_reference_points.unsqueeze(-2)], -2)
        self.track_instances.hist_velo =  torch.cat([self.track_instances.hist_velo[:, :, 1:], rec_velo.unsqueeze(-2)], -2)
        self.track_instances.hist_query_feats =  torch.cat([self.track_instances.hist_query_feats[:, :, 1:], rec_memory.unsqueeze(-2)], -2)
        self.track_instances.hist_mask[..., -1] = 1


    def post_merge_instances(self, data, kept_indicator=0): 
        """During training, we kept all activate instances, so the mergeing part should be after the assignment.
        """
        active_instances = (self.track_instances.matched_gt_idxes>=kept_indicator).nonzero()

        B = len(self.track_instances)
        topk_indexes_list = []
        for i in range(B):

            active_idxes_i = active_instances[active_instances[:, 0] == i][:, 1]
            scores = self.track_instances.scores[i:i+1].clone()
            scores[:, active_idxes_i] = -1
            
            _, topk_indexes = torch.topk(scores, self.topk_proposals - min(len(active_idxes_i), self.topk_proposals), dim=1)

            self.track_instances.obj_idxes[i, topk_indexes[0, :, 0]] = -1
            topk_indexes_list.append(torch.cat([active_idxes_i[None, :, None], topk_indexes], 1))

        topk_indexes = torch.cat(topk_indexes_list)
        # valid_key_set = ['reference_points', 'query_pos', 'query_feats', 'timestamp', 'velo', 'ego_pose', 'obj_idxes', 'matched_gt_idxes', 'disappear_time']
        topk_instances = self.track_instances.instances_topk_gather(topk_indexes, valid_key_set=None)
        re_track_instances = Instances.detach(topk_instances)
        self.history_track_instances = Instances.cat([re_track_instances, self.history_track_instances], dim=1)
        # self.memory_reference_point_copy = self.memory_reference_point.clone()

        self.history_track_instances.reference_points = transform_reference_points(self.history_track_instances.reference_points, data['ego_pose'], reverse=False)
        self.history_track_instances.timestamp -= data['timestamp'].unsqueeze(-1).unsqueeze(-1)
        self.history_track_instances.ego_pose = data['ego_pose'].unsqueeze(1) @ self.history_track_instances.ego_pose
    
        self.history_track_instances.hist_xyz = transform_reference_points(self.history_track_instances.hist_xyz, data['ego_pose'], reverse=False)
        self.history_track_instances.hist_velo = transform_velo(self.history_track_instances.hist_velo, data['ego_pose'], reverse=False)
        return topk_instances

    def forward(self, input_dict, img_metas,  gt_bboxes_3d=None, gt_labels_3d=None, debug_info=None):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """

        start_of_sequence = torch.FloatTensor([
            single_img_metas['start_of_sequence'] 
            for single_img_metas in img_metas]).to(input_dict['img_bev_feat'][0].device)

        timestamp = torch.FloatTensor([
            single_img_metas['timestamp'] 
            for single_img_metas in img_metas]).to(input_dict['img_bev_feat'][0].device)

        ego_pose_inv = torch.stack([
            single_img_metas['ego_pose_inv'] 
            for single_img_metas in img_metas], 0).to(input_dict['img_bev_feat'][0].device)

        ego_pose = torch.stack([
            single_img_metas['ego_pose'] 
            for single_img_metas in img_metas], 0).to(input_dict['img_bev_feat'][0].device)

        data = dict(
            start_of_sequence = start_of_sequence,
            timestamp = timestamp,
            ego_pose_inv = ego_pose_inv,
            ego_pose = ego_pose,
        )

        if input_dict['img_bev_feat'][0].dim() == 5:
            mlvl_feats = [level.mean(-1) for level in input_dict['img_bev_feat']]
        else:
            mlvl_feats = input_dict['img_bev_feat']

        # self.pre_update_memory(data)
        self.pre_update_instances(data)
        # mlvl_feats = data['img_feats']
        B = mlvl_feats[0].size(0)


        # reference_points = self.reference_points.weight
        dtype = self.track_instances.reference_points.dtype

        feat_flatten = []
        spatial_flatten = []
        for i in range(len(mlvl_feats)):
            B, C, H, W = mlvl_feats[i].shape
            mlvl_feat = mlvl_feats[i].reshape(B, C, -1).transpose(1, 2)
            # mlvl_feat = self.spatial_alignment(mlvl_feat, mln_input)
            feat_flatten.append(mlvl_feat.to(dtype))
            spatial_flatten.append((H, W))
        feat_flatten = torch.cat(feat_flatten, dim=1)
        spatial_flatten = torch.as_tensor(spatial_flatten, dtype=torch.long, device=mlvl_feats[0].device)
        level_start_index = torch.cat((spatial_flatten.new_zeros((1, )), spatial_flatten.prod(1).cumsum(0)[:-1]))
        # reference_points, attn_mask, mask_dict = self.prepare_for_dn(B, reference_points, img_metas,  gt_bboxes_3d, gt_labels_3d)
        attn_mask, mask_dict = None, None
        # prepare for the tgt and query_pos using mln.
        reference_points, temp_history_track_instances, temp_reference_points, rec_ego_pose = self.instance_temporal_alignment()

        tgt = self.track_instances.query_feats
        query_pos = self.track_instances.query_pos
        # reference_points = self.track_instances.reference_points
        temp_pos = temp_history_track_instances.query_pos
        temp_memory = temp_history_track_instances.query_feats
        

        outs_dec, intermediate_reference_points = self.transformer(tgt, query_pos, feat_flatten, spatial_flatten, level_start_index, temp_memory, 
                                    temp_pos, attn_mask, reference_points, self.pc_range, data, img_metas, reg_branches=self.reg_branches,
                                    return_intermediate_pts=True,
                                    query_embedding=self.query_embedding,
                                    temp_reference_points=temp_reference_points)

        outs_dec = torch.nan_to_num(outs_dec)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(outs_dec.shape[0]):
            reference = inverse_sigmoid(intermediate_reference_points[lvl])
            assert reference.shape[-1] == 3
            outputs_class = self.cls_branches[lvl](outs_dec[lvl])
            tmp = self.reg_branches[lvl](outs_dec[lvl])
            tmp[..., 0:3] += reference[..., 0:3]
            tmp[..., 0:3] = tmp[..., 0:3].sigmoid()
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        all_cls_scores = torch.stack(outputs_classes)
        all_bbox_preds = torch.stack(outputs_coords)
        all_bbox_preds[..., 0:3] = (all_bbox_preds[..., 0:3] * (self.pc_range[3:6] - self.pc_range[0:3]) + self.pc_range[0:3])
        
        # update the memory bank
        self.post_update_instances(data, rec_ego_pose, all_cls_scores, all_bbox_preds, outs_dec, mask_dict)

        if mask_dict and mask_dict['pad_size'] > 0:
            assert False
            output_known_class = all_cls_scores[:, :, :mask_dict['pad_size'], :]
            output_known_coord = all_bbox_preds[:, :, :mask_dict['pad_size'], :]
            outputs_class = all_cls_scores[:, :, mask_dict['pad_size']:, :]
            outputs_coord = all_bbox_preds[:, :, mask_dict['pad_size']:, :]
            mask_dict['output_known_lbs_bboxes']=(output_known_class, output_known_coord)
            outs = {
                'all_cls_scores': outputs_class,
                'all_bbox_preds': outputs_coord,
                'dn_mask_dict':mask_dict,
            }
        else:
            outs = {
                'agent_queries': self.track_instances.query_feats,
                'all_cls_scores': all_cls_scores,
                'all_bbox_preds': all_bbox_preds,
                'dn_mask_dict':None,
                'track_instances': self.track_instances,
                'data': data
            }

        return outs

   
    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             img_metas=None,
             gt_bboxes_ignore=None):
        """"Loss function.
        Args:
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indexes for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """


        instance_inds = [ single_img_metas['instance_inds'] for single_img_metas in img_metas]
        loss = self.criterion.loss_single_frame(0,
             gt_bboxes_list,
             gt_labels_list,
             instance_inds,
             preds_dicts,
             gt_bboxes_ignore)
        topk_instances = self.post_merge_instances(preds_dicts['data'])
        return loss, topk_instances
    
    def get_targets(self):
        pass
    
    def forward_tracking(self, input_dict, img_metas):
        pred_dicts = self.forward(input_dict, img_metas)
        # prev_active_track_instances = self.runtime_tracker.track_instances
        track_instances= pred_dicts['track_instances']

        # assign ids
        # active_mask = (track_instances.scores > self.runtime_tracker.threshold)
        B =  len(track_instances)
        appear_mask = (track_instances.obj_idxes< 0) & (track_instances.scores[..., 0] > self.runtime_tracker.threshold)        
        kept_mask = (track_instances.obj_idxes>=0) & (track_instances.scores[..., 0] > self.runtime_tracker.threshold)
        disappear_mask = (track_instances.obj_idxes>=0) & (track_instances.scores[..., 0] <= self.runtime_tracker.threshold)
        non_mask =  (track_instances.obj_idxes<0) & (track_instances.scores[..., 0] <= self.runtime_tracker.threshold)
        track_instances.matched_gt_idxes[appear_mask|kept_mask] = 1
        track_instances.matched_gt_idxes[disappear_mask] -= 1
        track_instances.matched_gt_idxes[non_mask] = -10000
        track_instances.obj_idxes[appear_mask] = torch.arange(self.runtime_tracker.current_id, self.runtime_tracker.current_id+appear_mask.sum(), device=appear_mask.device)[None]
        self.runtime_tracker.current_id += appear_mask.sum()
        
        pred_dicts['track_instances'] = track_instances.clone()
        pred_dicts['track_instances'].scores = pred_dicts['track_instances'].scores.squeeze(-1)
        score_mask = (pred_dicts['track_instances'].scores > self.runtime_tracker.output_threshold)
        pred_dicts['all_masks'] = score_mask.clone()

        topk_instances =  self.post_merge_instances(pred_dicts['data'], kept_indicator=0)


        return pred_dicts, topk_instances


    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        
        preds_dicts = self.bbox_coder.decode(preds_dicts, layer_index=self.layer_index)
        num_samples = len(preds_dicts)
       
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = img_metas[i]['box_type_3d'](bboxes, bboxes.size(-1))
            scores = preds['scores']
            labels = preds['labels']
            bbox_results = bbox3d2result(bboxes, scores, labels)
            for key in ['track_scores', 'obj_idxes']:
                bbox_results[key] = preds[key].cpu()
            ret_list.append(bbox_results)
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