# ------------------------------------------------------------------------
# Copyright (c) Toyota Research Institute
# ------------------------------------------------------------------------
# Modified from PETR (https://github.com/megvii-research/PETR)
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32
from mmdet.models import LOSSES
from mmdet.models import build_loss
from mmdet.core import (build_assigner, reduce_mean, multi_apply, build_sampler)
from mmdet3d.models.fbbev.track_head.streampetr_utils import normalize_bbox
from .tracking_loss import TrackingLoss


@LOSSES.register_module()
class TrackingLossCombo(TrackingLoss):
    """ Tracking loss with reference point supervision
    """
    def __init__(self,
                 num_classes,
                 code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
                 sync_cls_avg_factor=False,
                 interm_loss=True,
                 loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0),
                 loss_bbox=dict(type='L1Loss', loss_weight=0.25),
                 loss_iou=dict(type='GIoULoss', loss_weight=0.0),
                 loss_prediction=dict(type='L1Loss', loss_weight=1.0),
                 assigner=dict(
                    type='HungarianAssigner3D',
                    cls_cost=dict(type='FocalLossCost', weight=2.0),
                    reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                    iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head. 
                    pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])):

        super(TrackingLoss, self).__init__(
            num_classes, code_weights, sync_cls_avg_factor, interm_loss,
            loss_cls, loss_bbox, loss_iou, assigner)
        self.loss_traj = build_loss(loss_prediction)
        self.loss_mem_cls = build_loss(loss_cls)
        # self.loc_refine_code_weights = [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.loc_refine_code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
    
    def loss_prediction(self,
                        frame_idx,
                        loss_dict,
                        gt_trajs,
                        gt_masks,
                        pred_trajs,
                        loss_key='for'):
        loss_prediction = self.loss_traj(
            gt_trajs[..., :2] * gt_masks.unsqueeze(-1), 
            pred_trajs[..., :2] * gt_masks.unsqueeze(-1))
        loss_dict[f'f{frame_idx}.loss_{loss_key}'] = loss_prediction
        return loss_dict
    
    def loss_mem_bank(self,
                      frame_idx,
                      loss_dict,
                      gt_bboxes_list,
                      gt_labels_list,
                      instance_inds,
                      track_instances):
        obj_idxes_list = instance_inds[0].detach().cpu().numpy().tolist()
        obj_idx_to_gt_idx = {obj_idx: gt_idx for gt_idx, obj_idx in enumerate(obj_idxes_list)}
        device = track_instances.query_feats.device

        # classification loss
        matched_labels = torch.ones((len(track_instances), ), dtype=torch.long, device=device) * self.num_classes
        matched_label_weights = torch.ones((len(track_instances), ), dtype=torch.float32, device=device)
        num_pos, num_neg = 0, 0
        for track_idx, id in enumerate(track_instances.obj_idxes):
            cpu_id = id.cpu().numpy().tolist()
            if cpu_id not in obj_idx_to_gt_idx.keys():
                num_neg += 1
                continue
            index = obj_idx_to_gt_idx[cpu_id]
            matched_labels[track_idx] = gt_labels_list[0][index].long()
            num_pos += 1

        labels_list = matched_labels
        label_weights_list = matched_label_weights
        cls_scores = track_instances.cache_logits

        cls_avg_factor = num_pos * 1.0 + \
            num_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_mem_cls(
            cls_scores, labels_list, label_weights_list, avg_factor=cls_avg_factor)
        loss_cls = torch.nan_to_num(loss_cls)

        loss_dict[f'f{frame_idx}.loss_mem_cls'] = loss_cls

        # location refinement loss
        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        pos_bbox_num = 0
        matched_bbox_targets = torch.zeros((len(track_instances), gt_bboxes_list[0].shape[1]), dtype=torch.float32, device=device)
        matched_bbox_weights = torch.zeros((len(track_instances),len(self.loc_refine_code_weights)), dtype=torch.float32, device=device)
        for track_idx, id in enumerate(track_instances.obj_idxes):
            cpu_id = id.cpu().numpy().tolist()
            if cpu_id not in obj_idx_to_gt_idx.keys():
                matched_bbox_weights[track_idx] = 0.0
                continue
            index = obj_idx_to_gt_idx[cpu_id]
            matched_bbox_targets[track_idx] = gt_bboxes_list[0][index].float()
            matched_bbox_weights[track_idx] = 1.0
            pos_bbox_num += 1

        normalized_bbox_targets = normalize_bbox(matched_bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = matched_bbox_weights * torch.tensor(self.loc_refine_code_weights).to(device)

        loss_bbox = self.loss_bbox(
                track_instances.cache_bboxes[isnotnan, :10], normalized_bbox_targets[isnotnan, :10], bbox_weights[isnotnan, :10], avg_factor=pos_bbox_num)
        loss_dict[f'f{frame_idx}.loss_mem_bbox'] = loss_bbox
        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def forward(self,
                preds_dicts):
        """Loss function for multi-frame tracking
        """
        frame_num = len(preds_dicts)
        losses_dicts = [p.pop('loss_dict') for p in preds_dicts]
        loss_dict = dict()

        for key in losses_dicts[-1].keys():
            # example loss_dict["d2.loss_cls"] = losses_dicts[-1]["f0.d2.loss_cls"]
            loss_dict[key[3:]] = losses_dicts[-1][key]
        
        for frame_loss in losses_dicts[:-1]:
            loss_dict.update(frame_loss)

        return loss_dict


def nan_to_num(x, nan=0.0, posinf=None, neginf=None):
    x[torch.isnan(x)]= nan
    if posinf is not None:
        x[torch.isposinf(x)] = posinf
    if neginf is not None:
        x[torch.isneginf(x)] = posinf
    return x