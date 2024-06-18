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
from mmdet.models import DETECTORS
from mmdet3d.models import builder
from mmdet3d.models.detectors import CenterPoint
from mmdet3d.models.builder import build_head, build_neck
import numpy as np
import torch
import torchvision
import matplotlib
import cv2
import mmcv
from ..utils.grid_mask import GridMask
from ..utils.bricks import save_tensor

def generate_forward_transformation_matrix(bda, img_meta_dict=None):
    b = bda.size(0)
    hom_res = torch.eye(4)[None].repeat(b, 1, 1).to(bda.device)
    for i in range(b):
        hom_res[i, :3, :3] = bda[i]
    return hom_res


@DETECTORS.register_module()
class BEVPlanner(CenterPoint):

    def __init__(self, 
                 # BEVDet components
                 img_bev_encoder_backbone=None,
                 img_bev_encoder_neck=None,
                 forward_projection=None,
                 # BEVFormer components
                 backward_projection=None,
                 # FB-BEV components
                 frpn=None,
                 # other modules
                 depth_net=None,
                 occupancy_head=None,
                 img_det_2d_head=None,
                 map_head=None,
                 motion_head=None,
                 planner_head=None,
                 # other settings.
                 use_depth_supervision=False,
                 add_forward_backbward_feats=False,
                 fix_void=False,
                 occupancy_save_path=None,
                 do_history=True,
                 interpolation_mode='bilinear',
                 fuse_history_bev=True,
                 history_cat_num=16,
                 history_cat_conv_out_channels=None,
                 embed_dims=80,
                 single_bev_num_channels=80,
                 use_grid_mask=False,
                 yolox_use_ml_feats=False,
                 with_ego_status=False,
                 align_prev_bev=True,
                  **kwargs):
        """
            Parameters:
                img_bev_encoder_backbone - 
                img_bev_encoder_neck - 
                forward_projection - 
                backward_projection -
                frpn - foreground region proposal network, used in FB-BEV
                depth_net -
                occupancy_head -
                img_det_2d_head -
                map_head -
                motion_head -
                planner_head -
                
                use_depth_supervision -
                add_forward_backbward_feats -
                fix_void - Used to fix legacy bugs in Occupancy
                occupancy_save_path -
                do_history - A Flag to start the temporal traning at i-th epoch
                interpolation_mode -
                fuse_history_bev - Weather to use history bev, which is different from `do_hisitory` 
                history_cat_num -
                history_cat_conv_out_channels -
                single_bev_num_channels -
                use_grid_mask -
                yolox_use_ml_feats -
                with_ego_status -
        """
        super(BEVPlanner, self).__init__(**kwargs)
        self.fix_void = fix_void
      
        # BEVDet init
        self.forward_projection = builder.build_neck(forward_projection) if forward_projection else None
        self.img_bev_encoder_backbone = builder.build_backbone(img_bev_encoder_backbone) if img_bev_encoder_backbone else None
        self.img_bev_encoder_neck = builder.build_neck(img_bev_encoder_neck) if img_bev_encoder_neck else None

        # BEVFormer init
        self.backward_projection = builder.build_head(backward_projection) if backward_projection else None
    
        # FB-BEV init
        if not self.forward_projection: assert not frpn, 'frpn relies on LSS'
        self.frpn = builder.build_head(frpn) if frpn else None

        # Depth Net
        self.depth_net = builder.build_head(depth_net) if depth_net else None

        # Occupancy Head
        self.occupancy_head = builder.build_head(occupancy_head) if occupancy_head else None

        # 2D det head
        self.img_det_2d_head = builder.build_head(img_det_2d_head) if img_det_2d_head else None

        # map head
        if map_head:
            map_head['train_cfg'] = kwargs.get('train_cfg', None)
            self.map_head = builder.build_head(map_head)
        else: 
            self.map_head = None

        # motion
        self.motion_head = builder.build_head(motion_head) if motion_head else None        

        # planner
        self.planner_head = builder.build_head(planner_head) if planner_head else None
        
        self.embed_dims = embed_dims

        self.use_grid_mask = use_grid_mask
        if self.use_grid_mask:
            self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        
        self.add_forward_backbward_feats = add_forward_backbward_feats # fuse voxel features and bev features
        self.use_depth_supervision = use_depth_supervision
        self.yolox_use_ml_feats = yolox_use_ml_feats
        self.occupancy_save_path = occupancy_save_path # for saving data\for submitting to test server


        self.with_ego_status = with_ego_status
        if self.with_ego_status:
            self.can_bus_mlp = nn.Sequential(
                nn.Linear(9, self.embed_dims // 2),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dims // 2, self.embed_dims),
                nn.ReLU(inplace=True),
                nn.LayerNorm(self.embed_dims)
            )
        # Deal with history
        self.single_bev_num_channels = single_bev_num_channels
        self.do_history = do_history
        self.interpolation_mode = interpolation_mode
        self.history_cat_num = history_cat_num
        self.history_cam_sweep_freq = 0.5 # seconds between each frame
        self.history_cat_conv_out_channels = history_cat_conv_out_channels
        self.align_prev_bev=align_prev_bev
        self.fuse_history_bev = fuse_history_bev
        if self.fuse_history_bev:
            self._init_fuse_layers()
        self.history_sweep_time = None
        self.history_bev = None
        self.history_bev_before_encoder = None
        self.history_seq_ids = None
        self.history_forward_augs = None

    def _init_fuse_layers(self):
        history_cat_conv_out_channels = (self.history_cat_conv_out_channels 
                                         if self.history_cat_conv_out_channels is not None 
                                         else self.single_bev_num_channels)
        ## Embed each sample with its relative temporal offset with current timestep
    
        conv = nn.Conv2d if self.forward_projection.nx[-1] == 1 else nn.Conv3d
        self.history_keyframe_time_conv = nn.Sequential(
             conv(self.single_bev_num_channels + 1,
                     self.single_bev_num_channels,
                     kernel_size=1,
                     padding=0,
                     stride=1),
             nn.SyncBatchNorm(self.single_bev_num_channels),
             nn.ReLU(inplace=True))
        ## Then concatenate and send them through an MLP.
        self.history_keyframe_cat_conv = nn.Sequential(
            conv(self.single_bev_num_channels * (self.history_cat_num + 1),
                    history_cat_conv_out_channels,
                    kernel_size=1,
                    padding=0,
                    stride=1),
            nn.SyncBatchNorm(history_cat_conv_out_channels),
            nn.ReLU(inplace=True))



    def with_specific_component(self, component_name):
        """Whether the model owns a specific component"""
        return getattr(self, component_name, None) is not None
    
    def image_encoder(self, img):
        """
        Return (single_scale_context, multi_scale_context:[List])
        
        single scale_context are counsumed by forward projection
        multi_scale_context are consumed by some perception heads like yolox
        """
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        if self.use_grid_mask:
            imgs = self.grid_mask(imgs)
        x = self.img_backbone(imgs)
        
        if self.with_img_neck:
            x_list = self.img_neck(x)
            if type(x_list) in [list, tuple]:
                x_list = list(x_list)
                for i, x in enumerate(x_list):
                    _, output_dim, ouput_H, output_W = x.shape
                    x_list[i] = x.view(B, N, output_dim, ouput_H, output_W)
                return x_list[1], x_list
            else:
                _, output_dim, ouput_H, output_W = x_list.shape
                return x_list.view(B, N, output_dim, ouput_H, output_W), [x_list.view(B, N, output_dim, ouput_H, output_W)]
        

    @force_fp32()
    def bev_encoder(self, x):
        if self.with_specific_component('img_bev_encoder_backbone'):
            x = self.img_bev_encoder_backbone(x)
        
        if self.with_specific_component('img_bev_encoder_neck'):
            x = self.img_bev_encoder_neck(x)
        
        if type(x) not in [list, tuple]:
             x = [x]

        return x
    
    @force_fp32()
    def fuse_history(self, curr_bev, img_metas, bda): # align features with 3d shift

        if curr_bev is None: return None
        voxel_feat = True  if len(curr_bev.shape) == 5 else False
        if voxel_feat:
            curr_bev = curr_bev.permute(0, 1, 4, 2, 3) # n, c, z, h, w
        
        seq_ids = torch.LongTensor([
            single_img_metas['sequence_group_idx'] 
            for single_img_metas in img_metas]).to(curr_bev.device)
        start_of_sequence = torch.BoolTensor([
            single_img_metas['start_of_sequence'] 
            for single_img_metas in img_metas]).to(curr_bev.device)
        forward_augs = generate_forward_transformation_matrix(bda)
      
        # print('sqe_ids', seq_ids, ' start_of_sequence ', start_of_sequence.tolist(), ' index ', img_metas[0]['index'], img_metas[0]['scene_name'])

        curr_to_prev_ego_rt = torch.stack([
            single_img_metas['curr_to_prev_ego_rt']
            for single_img_metas in img_metas]).to(curr_bev)

        if not self.align_prev_bev:
            curr_to_prev_ego_rt= torch.eye(4).repeat(curr_to_prev_ego_rt.size(0), 1, 1).to(curr_bev)

        ## Deal with first batch
        if self.history_bev is None:
            self.history_bev = curr_bev.clone()
            self.history_seq_ids = seq_ids.clone()
            self.history_forward_augs = forward_augs.clone()

            # Repeat the first frame feature to be history
            if voxel_feat:
                self.history_bev = curr_bev.repeat(1, self.history_cat_num, 1, 1, 1) 
            else:
                self.history_bev = curr_bev.repeat(1, self.history_cat_num, 1, 1)
            # All 0s, representing current timestep.
            self.history_sweep_time = curr_bev.new_zeros(curr_bev.shape[0], self.history_cat_num)


        self.history_bev = self.history_bev.detach()

        assert self.history_bev.dtype == torch.float32

        ## Deal with the new sequences
        # First, sanity check. For every non-start of sequence, history id and seq id should be same.

        assert (self.history_seq_ids != seq_ids)[~start_of_sequence].sum() == 0, \
                "{}, {}, {}".format(self.history_seq_ids, seq_ids, start_of_sequence)

        ## Replace all the new sequences' positions in history with the curr_bev information
        self.history_sweep_time += 1 # new timestep, everything in history gets pushed back one.
        if start_of_sequence.sum()>0:
            if voxel_feat:    
                self.history_bev[start_of_sequence] = curr_bev[start_of_sequence].repeat(1, self.history_cat_num, 1, 1, 1)
            else:
                self.history_bev[start_of_sequence] = curr_bev[start_of_sequence].repeat(1, self.history_cat_num, 1, 1)
            
            self.history_sweep_time[start_of_sequence] = 0 # zero the new sequence timestep starts
            self.history_seq_ids[start_of_sequence] = seq_ids[start_of_sequence]
            self.history_forward_augs[start_of_sequence] = forward_augs[start_of_sequence]

        ## Get grid idxs & grid2bev first.
        if voxel_feat:
            n, c_, z, h, w = curr_bev.shape
        else:
            n, c_, h, w = curr_bev.shape
            z = 1

        # Generate grid
        xs = torch.linspace(0, w - 1, w, dtype=curr_bev.dtype, device=curr_bev.device).view(1, w, 1).expand(h, w, z)
        ys = torch.linspace(0, h - 1, h, dtype=curr_bev.dtype, device=curr_bev.device).view(h, 1, 1).expand(h, w, z)
        zs = torch.linspace(0, z - 1, z, dtype=curr_bev.dtype, device=curr_bev.device).view(1, 1, z).expand(h, w, z)
        grid = torch.stack(
            (xs, ys,  zs, torch.ones_like(xs)), -1).view(1, h, w, z, 4).expand(n, h, w, z, 4).view(n, h, w, z, 4, 1)

        # This converts BEV indices to meters
        # IMPORTANT: the feat2bev[0, 3] is changed from feat2bev[0, 2] because previous was 2D rotation
        # which has 2-th index as the hom index. Now, with 3D hom, 3-th is hom
        feat2bev = torch.zeros((4,4),dtype=grid.dtype).to(grid)
        feat2bev[0, 0] = self.forward_projection.dx[0]
        feat2bev[1, 1] = self.forward_projection.dx[1]
        feat2bev[2, 2] = self.forward_projection.dx[2]
        feat2bev[0, 3] = self.forward_projection.bx[0] - self.forward_projection.dx[0] / 2.
        feat2bev[1, 3] = self.forward_projection.bx[1] - self.forward_projection.dx[1] / 2.
        feat2bev[2, 3] = self.forward_projection.bx[2] - self.forward_projection.dx[2] / 2.
        feat2bev[3, 3] = 1
        feat2bev = feat2bev.view(1,4,4)
        
        ## Get flow for grid sampling.
        # The flow is as follows. Starting from grid locations in curr bev, transform to BEV XY11,
        # backward of current augmentations, curr lidar to prev lidar, forward of previous augmentations,
        # transform to previous grid locations.
        rt_flow = (torch.inverse(feat2bev) @ self.history_forward_augs @ curr_to_prev_ego_rt
                   @ torch.inverse(forward_augs) @ feat2bev)
        grid = rt_flow.view(n, 1, 1, 1, 4, 4) @ grid
        

        # normalize and sample
        if voxel_feat:
            normalize_factor = torch.tensor([w - 1.0, h - 1.0, z - 1.0], dtype=curr_bev.dtype, device=curr_bev.device)
            grid = grid[:,:,:,:, :3,0] / normalize_factor.view(1, 1, 1, 1, 3) * 2.0 - 1.0
        else:
            normalize_factor = torch.tensor([w - 1.0, h - 1.0], dtype=curr_bev.dtype, device=curr_bev.device)
            grid = grid[:,:,:,:, :2,0] / normalize_factor.view(1, 1, 1, 1, 2) * 2.0 - 1.0           

        tmp_bev = self.history_bev
        if voxel_feat: 
            n, mc, z, h, w = tmp_bev.shape
            tmp_bev = tmp_bev.reshape(n, mc, z, h, w)
            grid = grid.to(curr_bev.dtype).permute(0, 3, 1, 2, 4)
        else:
            grid = grid.to(curr_bev.dtype).squeeze(-2)

        # save_tensor(tmp_bev[0].clamp(min=-1, max=1).reshape(4, 80, 128, 128).abs().mean(1), f'curr_{self.count}_pre.png')
        sampled_history_bev = F.grid_sample(tmp_bev, grid, align_corners=True, mode=self.interpolation_mode)
        # save_tensor(sampled_history_bev[0].clamp(min=-1, max=1).reshape(4, 80, 128, 128).abs().mean(1), f'curr_{self.count}_after.png')
        # save_tensor(curr_bev.clamp(min=-1, max=1).abs().mean(1), f'curr_{self.count}.png')
        # self.count += 1
        # if self.count == 10:

        ## Update history
        # Add in current frame to features & timestep
        self.history_sweep_time = torch.cat(
            [self.history_sweep_time.new_zeros(self.history_sweep_time.shape[0], 1), self.history_sweep_time],
            dim=1) # B x (1 + T)

        if voxel_feat:
            sampled_history_bev = sampled_history_bev.reshape(n, mc, z, h, w)
            curr_bev = curr_bev.reshape(n, c_, z, h, w)
        feats_cat = torch.cat([curr_bev, sampled_history_bev], dim=1) # B x (1 + T) * 80 x H x W or B x (1 + T) * 80 xZ x H x W 

        # Reshape and concatenate features and timestep
        feats_to_return = feats_cat.reshape(
                feats_cat.shape[0], self.history_cat_num + 1, self.single_bev_num_channels, *feats_cat.shape[2:]) # B x (1 + T) x 80 x H x W
        if voxel_feat:
            feats_to_return = torch.cat(
            [feats_to_return, self.history_sweep_time[:, :, None, None, None, None].repeat(
                1, 1, 1, *feats_to_return.shape[3:]) * self.history_cam_sweep_freq
            ], dim=2) # B x (1 + T) x 81 x Z x H x W
        else:
            feats_to_return = torch.cat(
            [feats_to_return, self.history_sweep_time[:, :, None, None, None].repeat(
                1, 1, 1, feats_to_return.shape[3], feats_to_return.shape[4]) * self.history_cam_sweep_freq
            ], dim=2) # B x (1 + T) x 81 x H x W

        # Time conv
        feats_to_return = self.history_keyframe_time_conv(
            feats_to_return.reshape(-1, *feats_to_return.shape[2:])).reshape(
                feats_to_return.shape[0], feats_to_return.shape[1], -1, *feats_to_return.shape[3:]) # B x (1 + T) x 80 xZ x H x W

        # Cat keyframes & conv
        feats_to_return = self.history_keyframe_cat_conv(
            feats_to_return.reshape(
                feats_to_return.shape[0], -1, *feats_to_return.shape[3:])) # B x C x H x W or B x C x Z x H x W
        
        self.history_bev = feats_cat[:, :-self.single_bev_num_channels, ...].detach().clone()
        self.history_sweep_time = self.history_sweep_time[:, :-1]
        self.history_forward_augs = forward_augs.clone()
        if voxel_feat:
            feats_to_return = feats_to_return.permute(0, 1, 3, 4, 2)
        if not self.do_history:
            self.history_bev = None
        return feats_to_return.clone()


    def extract_img_bev_feat(self, img, img_metas, **kwargs):
        """Extract features of images."""

        return_map = {}

        context, mlvl_context = self.image_encoder(img[0])

        cam_params = img[1:7]
        if self.with_specific_component('depth_net'):
            mlp_input = self.depth_net.get_mlp_input(*cam_params)
            context, depth = self.depth_net(context, mlp_input)
        else:
            depth=None
                

        if self.with_specific_component('forward_projection'):
            bev_feat = self.forward_projection(cam_params, context, depth, **kwargs)
        else:
            bev_feat = None
        
        if self.with_specific_component('frpn'): # not used in FB-OCC
            assert bev_feat is not None
            bev_mask_logit = self.frpn(bev_feat)
            bev_mask = bev_mask_logit.sigmoid() > self.frpn.mask_thre            
            if bev_mask.requires_grad: # during training phase
                gt_bev_mask = kwargs['gt_bev_mask'].to(torch.bool)
                bev_mask = gt_bev_mask | bev_mask
            return_map['bev_mask_logit'] = bev_mask_logit    
        else:
            bev_mask = None

        if self.with_specific_component('backward_projection'):
            bev_feat_refined = self.backward_projection([context],
                                        img_metas,
                                        lss_bev=bev_feat.mean(-1),
                                        cam_params=cam_params,
                                        bev_mask=bev_mask,
                                        gt_bboxes_3d=None, # debug
                                        pred_img_depth=depth)  
                                        
            if self.add_forward_backbward_feats:
                bev_feat = bev_feat_refined[..., None] + bev_feat
            else:
                bev_feat = bev_feat_refined

        # Fuse History
        if self.fuse_history_bev:
            bev_feat = self.fuse_history(bev_feat, img_metas, img[6])

        if self.with_ego_status:
            can_bus_info = torch.cat(kwargs['can_bus_info'])
            bev_feat = bev_feat + self.can_bus_mlp(can_bus_info)[:, :, None, None]

        bev_feat = self.bev_encoder(bev_feat)
        
        
        return_map['context'] = mlvl_context if self.yolox_use_ml_feats else context
        return_map['depth'] = depth
        return_map['cam_params'] = cam_params
        return_map['img_bev_feat'] = bev_feat

        return return_map

    def extract_lidar_bev_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""

        voxels, num_points, coors = self.voxelize(pts)

        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        bev_feat = self.pts_middle_encoder(voxel_features, coors, batch_size)
        bev_feat = self.pts_backbone(bev_feat)
        if self.with_pts_neck:
            bev_feat = self.pts_neck(bev_feat)
        bev_feat = self.bev_encoder(bev_feat)
        return dict(lidar_bev_feat=bev_feat)

    def extract_feat(self, points, img, img_metas, **kwargs):
        """Extract features from images and points."""
        results={}
        if img is not None and self.with_specific_component('image_encoder'):
            results.update(self.extract_img_bev_feat(img, img_metas, **kwargs))
        if points is not None and self.with_specific_component('pts_voxel_encoder'):
            results.update(self.extract_lidar_bev_feat(points, img, img_metas))

        return results

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """

        results= self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        losses = dict()

        if self.with_pts_bbox:
            preds_agent_dicts = self.pts_bbox_head(results, img_metas,  gt_bboxes_3d, gt_labels_3d)
            losses_pts, agent_instances = self.pts_bbox_head.loss(gt_bboxes_3d,
                                            gt_labels_3d, preds_agent_dicts, img_metas)
            losses.update(losses_pts)
        
        if self.with_specific_component('img_det_2d_head'):
            if type(results['context']) not in [list, tuple]:
                context = [results['context']]
            else: context = results['context']
            preds_2ddet_dicts = self.img_det_2d_head(context)
            losses.update(
                self.img_det_2d_head.loss(
                    kwargs['gt_bboxes_2d'],
                    kwargs['gt_labels_2d'],
                    kwargs['centers2d'],
                    preds_2ddet_dicts,
                    kwargs['depths2d'],
                    img_metas, #len=B
                )
            )  

        if self.with_specific_component('occupancy_head'):
            losses_occupancy = self.occupancy_head.forward_train(results['img_bev_feat'], results=results, gt_occupancy=kwargs['gt_occupancy'], gt_occupancy_flow=kwargs['gt_occupancy_flow'])
            losses.update(losses_occupancy)

        if self.with_specific_component('map_head'):
            loss_map_dict, preds_map_dicts = self.map_head.forward(results, img_metas,  kwargs['map_gt_bboxes_3d'], kwargs['map_gt_labels_3d'], return_loss=True)
            losses.update(loss_map_dict)
        else: preds_map_dicts = [None] # dummy

        if self.with_specific_component('frpn'):
            losses_mask = self.frpn.get_bev_mask_loss(kwargs['gt_bev_mask'], results['bev_mask_logit'])
            losses.update(losses_mask)

        if self.use_depth_supervision and self.with_specific_component('depth_net'):
            loss_depth = self.depth_net.get_depth_loss(kwargs['gt_depth'], results['depth'])
            losses.update(loss_depth)

        if self.with_specific_component('motion_head'):
            preds_motion_dicts = self.motion_head(
                agent_instances,
                preds_map_dicts[-1],
                gt_ego_lcf_feat = kwargs['gt_ego_lcf_feat'],
                gt_ego_fut_cmd = kwargs['gt_ego_fut_cmd'],
                gt_ego_his_traj = kwargs['gt_ego_his_trajs'],
                gt_ego_fut_trajs = kwargs['gt_ego_fut_trajs'],
                img_metas=img_metas,
            )
            losses.update(
                self.motion_head.loss(
                    gt_agent_fut_traj = kwargs['gt_agent_fut_traj'],
                    gt_agent_fut_traj_mask = kwargs['gt_agent_fut_traj_mask'],
                    gt_ego_fut_cmd = kwargs['gt_ego_fut_cmd'],
                    gt_ego_fut_trajs = kwargs['gt_ego_fut_trajs'],
                    gt_ego_fut_masks = kwargs['gt_ego_fut_masks'],
                    preds_dicts = preds_motion_dicts,
                    preds_map_dicts =  preds_map_dicts[-1],
                    matched_gt_idxes = agent_instances.matched_gt_idxes,
                    img_metas = img_metas,
                )
            )
        
        if self.with_specific_component('planner_head'):
            preds_plan_dicts = self.planner_head(
                results,
                kwargs['gt_ego_lcf_feat'],
                kwargs['gt_ego_fut_cmd'],
                kwargs['gt_ego_his_trajs'],
                kwargs['gt_ego_fut_trajs'],
                img_metas=img_metas,
                map_results=preds_map_dicts[-1]
            )
            losses.update(
                self.planner_head.loss(
                    kwargs['gt_ego_fut_trajs'],
                    kwargs['gt_ego_fut_cmd'],
                    kwargs['gt_ego_fut_masks'],
                    preds_plan_dicts,
                    img_metas,
                )
            )
        
        return losses

    def forward_test(self,
                     points=None,
                     img_metas=None,
                     img_inputs=None,
                     **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        self.do_history = True
        kwargs['can_bus_info'] = kwargs.get('can_bus_info', [None])[0]
        if img_inputs is not None:
            for var, name in [(img_inputs, 'img_inputs'),
                          (img_metas, 'img_metas')]:
                if not isinstance(var, list) :
                    raise TypeError('{} must be a list, but got {}'.format(
                        name, type(var)))        
            num_augs = len(img_inputs)
            if num_augs != len(img_metas):
                raise ValueError(
                    'num of augmentations ({}) != num of image meta ({})'.format(
                        len(img_inputs), len(img_metas)))

            if num_augs==1 and not img_metas[0][0].get('tta_config', dict(dist_tta=False))['dist_tta']:
                return self.simple_test(points[0], img_metas[0], img_inputs[0],
                                    **kwargs)
            else:
                return self.aug_test(points, img_metas, img_inputs, **kwargs)
        
        elif points is not None:
            img_inputs = [img_inputs] if img_inputs is None else img_inputs
            points = [points] if points is None else points
            return self.simple_test(points[0], img_metas[0], img_inputs[0],
                                    **kwargs)
        
    def aug_test(self,points,
                    img_metas,
                    img_inputs=None,
                    visible_mask=[None],
                    **kwargs):
        """Test function without augmentaiton."""
        assert False
        return None

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    visible_mask=[None],
                    return_raw_occ=False,
                    **kwargs):
        """Test function without augmentaiton."""
        results = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        
        output_list = [dict() for _ in range(len(img_metas))]
        
        if  self.with_pts_bbox:
            if getattr(self.pts_bbox_head, 'tracking', False):
                preds_det_dicts, agent_instances = self.pts_bbox_head.forward_tracking(results, img_metas)
            else:
                preds_det_dicts = self.pts_bbox_head(results, img_metas)
            pred_bbox = self.pts_bbox_head.get_bboxes(preds_det_dicts, img_metas, rescale=rescale)
            pred_bbox[0]['index'] =  img_metas[0]['index']
        else:
            pred_bbox = [None for _ in range(len(img_metas))]

        if self.with_specific_component('map_head'):
            preds_map_dicts = self.map_head(results,
             img_metas,
             return_loss=False,
             map_gt_bboxes_3d = kwargs.get('map_gt_bboxes_3d', None),
             map_gt_labels_3d = kwargs.get('map_gt_labels_3d', None),
             )
            pred_map = self.map_head.get_bboxes(preds_map_dicts, img_metas)
            pred_map[0]['index'] =  img_metas[0]['index']
        else:
            preds_map_dicts = [None] # dummy
            pred_map = [None for _ in range(len(img_metas))]

        if self.with_specific_component('motion_head'):
            preds_motion_dicts = self.motion_head(
                agent_instances,
                preds_map_dicts[-1],
                gt_ego_lcf_feat = kwargs['gt_ego_lcf_feat'][0],
                gt_ego_fut_cmd = kwargs['gt_ego_fut_cmd'][0],
                gt_ego_his_traj = kwargs['gt_ego_his_trajs'][0],
                gt_ego_fut_trajs = kwargs['gt_ego_fut_trajs'][0],
                img_metas=img_metas,
            )
            pred_motion = self.motion_head.get_motion(preds_motion_dicts, img_metas)
            pred_motion[0]['index'] =  img_metas[0]['index']
            pred_traj = self.motion_head.get_traj(
                preds_motion_dicts,
                img_metas,
                gt_ego_fut_trajs=kwargs['gt_ego_fut_trajs'][0],
                gt_ego_fut_cmd=kwargs['gt_ego_fut_cmd'][0],
                gt_ego_fut_masks=kwargs['gt_ego_fut_masks'][0],
                gt_fut_segmentations=kwargs['gt_fut_segmentations'][0],
                gt_fut_segmentations_plus=kwargs['gt_fut_segmentations_plus'][0],
                # vad_ego_fut_trajs=kwargs['vad_ego_fut_trajs'][0],
             )
            pred_traj[0]['index'] =  img_metas[0]['index']

            # add motion traj to tracking results
            num_bbox = pred_bbox[0]['track_scores'].size(0)
            motion_info = np.zeros([num_bbox, 6, 8, 2])
            motion_cls = np.zeros([num_bbox, 6])
            for i, obj_idx in enumerate(pred_motion[0]['obj_idxes']):
                try:
                    bbox_ind = (pred_bbox[0]['obj_idxes']==obj_idx).nonzero().item()
                except:
                    continue
                motion_info[bbox_ind] = pred_motion[0]['fut_trajs_in_global'][i]
                motion_cls[bbox_ind] = pred_motion[0]['pred_traj_cls'][i]
            pred_bbox[0]['motion_traj'] = motion_info
            pred_bbox[0]['motion_cls'] = motion_cls

        else:
            pred_motion = [None for _ in range(len(img_metas))]
            pred_traj = [None for _ in range(len(img_metas))]


        if self.with_specific_component('occupancy_head'):
            pred_occupancy = self.occupancy_head(results['img_bev_feat'], results=results, **kwargs)['output_voxels'][0]

            pred_occupancy = pred_occupancy.permute(0, 2, 3, 4, 1)[0]
            if self.fix_void:
                pred_occupancy = pred_occupancy[..., 1:]     
            pred_occupancy = pred_occupancy.softmax(-1)


            # convert to CVPR2023 Format
            pred_occupancy = pred_occupancy.permute(3, 2, 0, 1)
            pred_occupancy = torch.flip(pred_occupancy, [2])
            pred_occupancy = torch.rot90(pred_occupancy, -1, [2, 3])
            pred_occupancy = pred_occupancy.permute(2, 3, 1, 0)
            
            if return_raw_occ:
                pred_occupancy_category = pred_occupancy
            else:
                pred_occupancy_category = pred_occupancy.argmax(-1) 
            

            # # do not change the order
            # if self.occupancy_save_path is not None:
            #     scene_name = img_metas[0]['scene_name']
            #     sample_token = img_metas[0]['sample_idx']
            #     mask_camera = visible_mask[0][0]
            #     masked_pred_occupancy = pred_occupancy[mask_camera].cpu().numpy()
            #     save_path = os.path.join(self.occupancy_save_path, 'occupancy_pred', scene_name+'_'+sample_token)
            #     np.savez_compressed(save_path, pred=masked_pred_occupancy, sample_token=sample_token) 

            # For test server
            if self.occupancy_save_path is not None:
                    scene_name = img_metas[0]['scene_name']
                    sample_token = img_metas[0]['sample_idx']
                    # mask_camera = visible_mask[0][0]
                    # masked_pred_occupancy = pred_occupancy[mask_camera].cpu().numpy()
                    save_pred_occupancy = pred_occupancy.argmax(-1).cpu().numpy()
                    save_path = os.path.join(self.occupancy_save_path, 'occupancy_pred', f'{sample_token}.npz')
                    np.savez_compressed(save_path, save_pred_occupancy.astype(np.uint8)) 

            pred_occupancy_category= pred_occupancy_category.cpu().numpy()

        else:
            pred_occupancy_category =  None

        if self.with_specific_component('planner_head'):
            preds_dicts = self.planner_head(
                results,
                kwargs['gt_ego_lcf_feat'][0],
                kwargs['gt_ego_fut_cmd'][0],
                kwargs['gt_ego_his_trajs'][0],
                kwargs['gt_ego_fut_trajs'][0],
                img_metas=img_metas,
                map_results=preds_map_dicts[-1]
                )
            pred_traj = self.planner_head.get_bboxes(preds_dicts, img_metas, gt_ego_fut_trajs=kwargs['gt_ego_fut_trajs'][0],
             gt_ego_fut_cmd=kwargs['gt_ego_fut_cmd'][0], gt_ego_fut_masks=kwargs['gt_ego_fut_masks'][0], gt_fut_segmentations=kwargs['gt_fut_segmentations'][0],
             gt_fut_segmentations_plus=kwargs['gt_fut_segmentations_plus'][0],
             # vad_ego_fut_trajs=kwargs['vad_ego_fut_trajs'][0],
             )
            pred_traj[0]['index'] =  img_metas[0]['index']
        else:
            pred_traj = [None for _ in range(len(img_metas))]
        # if results.get('bev_mask_logit', None) is not None:
        #     pred_bev_mask = results['bev_mask_logit'].sigmoid() > 0.5

        assert len(img_metas) == 1
        for i, result_dict in enumerate(output_list):
            result_dict['pts_bbox'] = pred_bbox[i]
            result_dict['pred_map'] = pred_map[i]
            result_dict['pred_motion'] = pred_motion[i]
            result_dict['pred_ego_traj'] = pred_traj[i]
            result_dict['pred_occupancy'] = pred_occupancy_category
            result_dict['index'] = img_metas[i]['index']

        # if not self.training:
        #     self.visual_sample(output_list, **kwargs)
        
        return output_list


    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        results = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        assert self.with_pts_bbox
        outs = self.pts_bbox_head(results['img_bev_feat'])
        return outs

    def world2bev_vis(self, x, y):
             return int((x + 51.2) * 5), int((y + 51.2) * 5)

    def visual_sample(self, results, **kwargs):
        
        import cv2
        # upper image is gt
        bev_img = np.ones([1024, 512, 3], dtype=np.float32) * 255
        bev_img = bev_img.astype(np.float32)

        bev_img = cv2.circle(bev_img, self.world2bev_vis(0, 0), 5, (0, 255, 0), thickness=-1)
        bev_img = cv2.circle(bev_img, self.world2bev_vis(0, 51.2 * 2), 5, (0, 255, 0), thickness=-1)

        if results[0].get('pts_bbox') is not None:
            bbox = results[0]['pts_bbox']['boxes_3d']
            track_scores = results[0]['pts_bbox']['track_scores']
            for i, corners in enumerate(bbox.corners[:, [4, 7, 3, 0], :2]):
                if track_scores[i]<0.4: continue
                corners = np.array([self.world2bev_vis(*corner) for corner in corners])
                corners2 = np.array([(x, y+512) for (x, y) in corners])
                
                bev_img = cv2.circle(bev_img, corners[0], 1, (61, 102, 255))
                bev_img = cv2.polylines(bev_img, pts=[corners], isClosed=True, color=(61, 102, 255), thickness=1)
                
                bev_img = cv2.circle(bev_img, corners2[0], 1, (61, 102, 255))
                bev_img = cv2.polylines(bev_img, pts=[corners2], isClosed=True, color=(61, 102, 255), thickness=1)

        if kwargs.get('gt_bboxes_3d', False):
            gt_bboxes_3d = kwargs['gt_bboxes_3d'][0][0]
            for i, corners in enumerate(gt_bboxes_3d.corners[:, [4, 7, 3, 0], :2]):
                corners = np.array([self.world2bev_vis(*corner) for corner in corners])
                bev_img = cv2.circle(bev_img, corners[0], 1, (61, 102, 255))
                # bev_img = cv2.fillPoly(bev_img, [corners], (61, 102, 255))
                bev_img = cv2.polylines(bev_img, pts=[corners], isClosed=True, color=(255, 102, 61), thickness=1)



        if results[0].get('pred_ego_traj') is not None:
            pred_ego_fut_trajs = results[0]['pred_ego_traj']['pred_ego_fut_trajs']
            gt_ego_fut_trajs = results[0]['pred_ego_traj']['gt_ego_fut_trajs']

            gt_ego_fut_trajs, colors = self._render_traj(gt_ego_fut_trajs.numpy())
            points = np.array([self.world2bev_vis(*point) for point in gt_ego_fut_trajs])
            for point, color in zip(points, colors):
                bev_img = cv2.circle(bev_img, point, 1, color)

            pred_ego_fut_trajs, colors = self._render_traj(pred_ego_fut_trajs.numpy(), colormap='autumn')
            points = np.array([self.world2bev_vis(*point) for point in pred_ego_fut_trajs])
            for point, color in zip(points, colors):
                x,y = point
                bev_img = cv2.circle(bev_img, (x, y+512), 1, color)


        if kwargs.get('map_gt_bboxes_3d', False):
            map_gt_bboxes_3d = kwargs['map_gt_bboxes_3d'][0][0]
            map_gt_labels_3d = kwargs['map_gt_labels_3d'][0][0]
            for k, line in enumerate(map_gt_bboxes_3d.fixed_num_sampled_points):
                    label = map_gt_labels_3d[k]
                    # line = (line[..., :2] - self.map_head.origin.cpu()) / self.map_head.roi_size.cpu()
                    line = line.cpu().numpy()
                    corners = np.array([self.world2bev_vis(*corner) for corner in line])
                    corners = [each for each in corners if ((each>=0).all() & (each<512).all())]
                    colors = [(255, 255, 0), (255, 0, 0), (0, 255, 0)]
                    for i, corner in enumerate(corners[:-1]):
                        bev_img = cv2.circle(bev_img, corners[i], 2, (61, 102, 255))
                        bev_img = cv2.line(bev_img, corners[i], corners[i+1], color=colors[label], thickness=1)

        if results[0].get('pred_map') is not None:       
            for k, line in enumerate(results[0]['pred_map']['map_pts_3d']):
                label = results[0]['pred_map']['map_labels_3d'][k]
                # if label !=0: continue
                score = results[0]['pred_map']['map_scores_3d'][k]
                if score < 0.4: continue
                line = line.cpu().numpy()
                corners = np.array([self.world2bev_vis(*corner) for corner in line])
                corners = [each for each in corners if ((each>=0).all() & (each<512).all())]
                corners = [(x, y+512) for (x, y) in corners ]
                colors = [(255, 255, 0), (255, 0, 0), (0, 255, 0)]
                for i, corner in enumerate(corners[:-1]):
                    bev_img = cv2.circle(bev_img, corners[i], 2, (61, 102, 255))
                    bev_img = cv2.line(bev_img, corners[i], corners[i+1], color=colors[label], thickness=1)



        if kwargs.get('gt_agent_fut_traj', False):
            gt_agent_fut_traj = kwargs['gt_agent_fut_traj'][0][0].cpu()
            gt_agent_fut_traj_mask = kwargs['gt_agent_fut_traj_mask'][0][0].cpu()
            centers = kwargs['gt_bboxes_3d'][0][0].center[..., :2].cpu()
            tmp = torch.cat([centers[:, None], gt_agent_fut_traj], 1)
            trajs = torch.cumsum(tmp, 1)[:, 1:]
            for k, traj in enumerate(trajs):
                traj = traj.cpu().numpy()
                corners = np.array([self.world2bev_vis(*corner) for corner in traj])
                center = np.array(self.world2bev_vis(*centers[k]))
                corners = [each for each in corners if ((each>=0).all() & (each<1536).all())]
                colors = [(255, 255, 0), (255, 0, 0), (0, 255, 0)]
                for i, corner in enumerate(corners[:-1]):
                    if gt_agent_fut_traj_mask[k, i+1].sum()<2 or gt_agent_fut_traj_mask[k, i].sum()<2:
                        continue
                    if i == 0: 
                        bev_img = cv2.line(bev_img, center, corners[i], color=(123, 22, 187), thickness=1)
                    # bev_img = cv2.circle(bev_img, corners[i], 2, (61, 102, 32))
                    bev_img = cv2.line(bev_img, corners[i], corners[i+1], color=(123, 22, 187), thickness=1)

        
        if results[0].get('pred_motion') is not None:       
            
            obj_idxes_list = results[0]['pts_bbox']['obj_idxes']
            centers = results[0]['pts_bbox']['boxes_3d'].center[..., :2].cpu().numpy()
            
            # pred_agent_fut_trajs = results[0]['pred_motion']['pred_agent_fut_trajs']
            pred_agent_fut_trajs2 = results[0]['pred_motion']['pred_agent_fut_trajs2']
            motion_obj_idxes = results[0]['pred_motion']['obj_idxes']

            for k, trajs in enumerate(pred_agent_fut_trajs2):
                try:
                    track_k = (obj_idxes_list==motion_obj_idxes[k]).nonzero()[0][0]
                except:
                    continue
                if track_scores[track_k]<0.4: continue

                traj_ind = results[0]['pred_motion']['pred_traj_cls'][k].argmax()
                # for traj in trajs:
                traj = trajs[traj_ind]
                
                corners = np.array([self.world2bev_vis(*corner) for corner in traj])
                corners = np.array([(x, y+512) for (x, y) in corners])
                center = np.array(self.world2bev_vis(*centers[track_k]))
                center[-1] +=512
                corners = [each for each in corners if ((each>=0).all() & (each<1536).all())]
                colors = [(255, 255, 0), (255, 0, 0), (0, 255, 0)]
                for i, corner in enumerate(corners[:-1]):
                    if i == 0: bev_img = cv2.line(bev_img, center, corners[i], color=(123, 22, 187), thickness=1)
                    # bev_img = cv2.circle(bev_img, corners[i], 2, (61, 102, 32))
                    bev_img = cv2.line(bev_img, corners[i], corners[i+1], color=(22, 122, 187), thickness=1)

        mmcv.imwrite(bev_img, f'bev_{results[0]["index"]}.png')

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