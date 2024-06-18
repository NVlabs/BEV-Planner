# ------------------------------------------------------------------------
# Copyright (c) Toyota Research Institute
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
class TrackingLossPrediction(TrackingLoss):
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