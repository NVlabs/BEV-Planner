# Copyright (c) 2023-2024, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# TODO: add license here



# we follow the online training settings  from solofusion
num_gpus = 8
samples_per_gpu = 4
num_iters_per_epoch = int(28130 // (num_gpus * samples_per_gpu) )
num_epochs = 12
checkpoint_epoch_interval = 1
use_custom_eval_hook=True

# Each nuScenes sequence is ~40 keyframes long. Our training procedure samples
# sequences first, then loads frames from the sampled sequence in order 
# starting from the first frame. This reduces training step-to-step diversity,
# lowering performance. To increase diversity, we split each training sequence
# in half to ~20 keyframes, and sample these shorter sequences during training.
# During testing, we do not do this splitting.
train_sequences_split_num = 4
test_sequences_split_num = 1

# By default, 3D detection datasets randomly choose another sample if there is
# no GT object in the current sample. This does not make sense when doing
# sequential sampling of frames, so we disable it.
filter_empty_gt = False

# Long-Term Fusion Parameters
do_history = False
history_cat_num = 4
history_cat_conv_out_channels = 160

_base_ = ['../_base_/datasets/nus-3d.py', '../_base_/default_runtime.py']
# Global
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
# bev configs
roi_size = (102.4, 102.4)
bev_h = 128
bev_w = 128
point_cloud_range = [-roi_size[0]/2, -roi_size[1]/2, -5, roi_size[0]/2, roi_size[1]/2, 3]

# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams':
    6,
    'input_size': (256, 704),
    'src_size': (900, 1600),
    # Augmentation
    'resize': (0.38, 0.55),
    'rot': (0, 0),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}
bda_aug_conf = dict(
    rot_lim=(-0, 0),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.,
    flip_dy_ratio=0.)
voxel_size = [0.2, 0.2, 8]
use_checkpoint = False
sync_bn = True
# Model
grid_config = {
    'x': [-51.2, 51.2, 0.8],
    'y': [-51.2, 51.2, 0.8],
    'z': [-5, 3, 8],
    'depth': [1.0, 60.0, 0.5],
}
depth_categories = 118 #(grid_config['depth'][1]-grid_config['depth'][0])//grid_config['depth'][2]

numC_Trans=80
_dim_ = 256

### occupancy config
empty_idx = 18  # noise 0-->255
num_cls = 19  # 0 others, 1-16 obj, 17 free
fix_void = num_cls == 19
###

map_classes = ['divider', 'ped_crossing', 'boundary']
map_num_vec = 100
map_fixed_ptsnum_per_gt_line = 20 # now only support fixed_pts > 0
map_fixed_ptsnum_per_pred_line = 20
map_eval_use_same_gt_sample_num_flag = True
map_num_classes = len(map_classes)

embed_dims = 256
num_feat_levels = 1
norm_cfg = dict(type='BN2d')
num_queries = 100

# category configs
cat2id = {
    'ped_crossing': 0,
    'divider': 1,
    'boundary': 2,
}
num_class = max(list(cat2id.values())) + 1


num_points = 20
permute = True
with_ego_as_agent = False
###
model = dict(
    type='BEVPlanner',
    use_depth_supervision=False,
    fix_void=fix_void,
    do_history = do_history,
    history_cat_num=history_cat_num,
    single_bev_num_channels=numC_Trans,
    fuse_history_bev=True,
    use_grid_mask=True,
    align_prev_bev=False,
    img_backbone=dict(       
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        with_cp=True,
        pretrained='torchvision://resnet50',
        style='pytorch'),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[1024, 2048],
        out_channels=_dim_,
        num_outs=1,
        start_level=0,
        with_cp=use_checkpoint,
        out_ids=[0]),
    depth_net=dict(
        type='CM_DepthNet', # camera-aware depth net
        in_channels=_dim_,
        context_channels=numC_Trans,
        downsample=16,
        grid_config=grid_config,
        depth_channels=depth_categories,
        with_cp=use_checkpoint,
        loss_depth_weight=1.,
        aspp_mid_channels=96,
        use_dcn=False,
    ),
    forward_projection=dict(
        type='LSSViewTransformerFunction',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        downsample=16),
    frpn=None,
    backward_projection=None,
    img_bev_encoder_backbone=dict(
        type='CustomResNet',
        numC_input=numC_Trans,
        num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8]),
    img_bev_encoder_neck=dict(
        type='FPN_LSS',
        in_channels=numC_Trans * 8 + numC_Trans * 2,
        out_channels=256),
    occupancy_head=None,
    img_det_2d_head=None,
    pts_bbox_head=None,
    map_head=dict(
        type='MapDetectorHead',
        num_queries=num_queries,
        embed_dims=embed_dims,
        num_classes=num_class,
        in_channels=embed_dims,
        num_points=num_points,
        roi_size=roi_size,
        coord_dim=2,
        different_heads=False,
        predict_refine=False,
        sync_cls_avg_factor=True,
        streaming_cfg=dict(
            streaming=False,
            batch_size=samples_per_gpu,
            topk=int(num_queries*(1/3)),
            trans_loss_weight=0.1,
        ),
        # streaming_cfg=None,
        transformer=dict(
            type='MapTransformer',
            num_feature_levels=1,
            num_points=num_points,
            coord_dim=2,
            encoder=dict(
                type='PlaceHolderEncoder',
                embed_dims=embed_dims,
            ),
            decoder=dict(
                type='MapTransformerDecoder_new',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='MapTransformerLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=embed_dims,
                            num_heads=8,
                            attn_drop=0.1,
                            proj_drop=0.1,
                        ),
                        dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=embed_dims,
                            num_heads=8,
                            num_levels=1,
                            num_points=num_points,
                            dropout=0.1,
                        ),
                    ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=embed_dims,
                        feedforward_channels=embed_dims*2,
                        num_fcs=2,
                        ffn_drop=0.1,
                        act_cfg=dict(type='ReLU', inplace=True),        
                    ),
                    feedforward_channels=embed_dims*2,
                    ffn_dropout=0.1,
                    # operation_order=('norm', 'self_attn', 'norm', 'cross_attn',
                    #                 'norm', 'ffn',)
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                    'ffn', 'norm')
                )
            )
        ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=0.5
        ),
        loss_reg=dict(
            type='LinesL1Loss',
            loss_weight=5.0,
            beta=0.01,
        ),
        assigner=dict(
            type='HungarianLinesAssigner',
                cost=dict(
                    type='MapQueriesCost',
                    cls_cost=dict(type='FocalLossCost', weight=0.5),
                    reg_cost=dict(type='LinesL1Cost', weight=5.0, beta=0.01, permute=permute),
                    ),
                ),
        ),
    motion_head=None,
    planner_head=dict(
        type='NaivePlannerHead',
        use_map_info=True,
        loss_plan_reg=dict(type='L1Loss', loss_weight=20.0),
        loss_plan_col=dict(type='PlanCollisionLoss', loss_weight=20.0),
    ),
    # model training and testing settings
    train_cfg=dict(pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            assigner=None),
    )
)


# Data
dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')
occupancy_path = '/mount/data/occupancy_cvpr2023/gts'
normalize_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        normalize_cfg=normalize_cfg,
        data_config=data_config),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        with_2d_bbox=True,
        classes=class_names),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),  
    dict(
        type='LoadVectorMap2',
        data_root = data_root,
        point_cloud_range =point_cloud_range,
        map_classes = ['divider', 'ped_crossing', 'boundary'],
        map_num_vec = 100,
        map_fixed_ptsnum_per_line = 20, # now only support fixed_pts > 0,
        map_eval_use_same_gt_sample_num_flag = True,
        map_num_classes = 3,
    ),   
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(type='LoadGTMotion'),
    dict(type='LoadGTPlaner'),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    # dict(type='VisualInputsAndGT'),
    # dict(type='LoadOccupancy', ignore_nonvisible=True, fix_void=fix_void, occupancy_path=occupancy_path),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_depth', 'gt_bboxes_2d', 'gt_labels_2d', 'centers2d', 'depths2d',  'map_gt_labels_3d', 'map_gt_bboxes_3d'
                               ] + ['gt_agent_fut_traj', 'gt_agent_fut_traj_mask']+
                               ['gt_ego_lcf_feat', 'gt_ego_fut_trajs', 'gt_ego_his_trajs', 'gt_ego_fut_cmd', 'gt_ego_fut_masks']
                               )
]

test_pipeline = [
    dict(
        type='CustomDistMultiScaleFlipAug3D',
        tta=False,
        transforms=[
            dict(type='PrepareImageInputs',
            # img_corruptions='sun', 
            data_config=data_config, normalize_cfg=normalize_cfg),
            dict(
                type='LoadAnnotationsBEVDepth',
                bda_aug_conf=bda_aug_conf,
                classes=class_names,
                with_2d_bbox=True,
                
                is_train=False),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=file_client_args),
            dict(
                type='LoadVectorMap',
                data_root = data_root,
                point_cloud_range =point_cloud_range,
                map_classes = ['divider', 'ped_crossing', 'boundary'],
                map_num_vec = 100,
                map_fixed_ptsnum_per_line = 20, # now only support fixed_pts > 0,
                map_eval_use_same_gt_sample_num_flag = True,
                map_num_classes = 3,
            ),   
            dict(type='LoadGTPlaner'),
            dict(type='LoadGTMotion',  with_ego_as_agent=with_ego_as_agent),   
            dict(type='LoadFutBoxInfo'),
            dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
            dict(type='ObjectNameFilter', classes=class_names),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs', 'gt_bboxes_3d', 'gt_labels_3d', 'map_gt_bboxes_3d', 'map_gt_labels_3d']+
            ['gt_agent_fut_traj', 'gt_agent_fut_traj_mask']+['gt_ego_lcf_feat', 'gt_ego_fut_trajs', 'gt_ego_his_trajs', 'gt_ego_fut_cmd', 'gt_ego_fut_masks']+
            ['gt_fut_segmentations', 'gt_fut_segmentations_plus', 'fut_boxes_in_cur_ego_list']    
            )
            ]
        )
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

share_data_config = dict(
    type=dataset_type,
    classes=class_names,
    modality=input_modality,
    img_info_prototype='bevdet',
    occupancy_path=occupancy_path,
    data_root=data_root,
    use_sequence_group_flag=True,
)

test_data_config = dict(
    pipeline=test_pipeline,
    map_ann_file=data_root + 'nuscenes_map_infos_102x102_val.pkl',
    map_eval_cfg=dict(
        region = (102.4, 102.4) # (H, W)
    ),
    load_fut_bbox_info=True,
    sequences_split_num=test_sequences_split_num,
    ann_file=data_root + 'bev-next-nuscenes_infos_val.pkl')

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=2,
    test_dataloader=dict(runner_type='IterBasedRunnerEval'),
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'bev-next-nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        test_mode=False,
        use_valid_flag=True,
        
        sequences_split_num=train_sequences_split_num,
        filter_empty_gt=filter_empty_gt,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=test_data_config,
    test=test_data_config)

for key in ['train', 'val', 'test']:
    data[key].update(share_data_config)


optimizer = dict(
    type='AdamW', 
    lr=1e-4, # bs 8: 2e-4 || bs 16: 4e-4
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1), 
        }),
    weight_decay=0.01)
 
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
    )

runner = dict(type='IterBasedRunner', max_iters=num_epochs * num_iters_per_epoch)
checkpoint_config = dict(
    interval=checkpoint_epoch_interval * num_iters_per_epoch)
evaluation = dict(
    interval=num_epochs * num_iters_per_epoch, pipeline=test_pipeline)


log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
        interval=2*num_iters_per_epoch,
    ),
    dict(
        type='SequentialControlHook',
        temporal_start_iter=0,
    ),
    # dict(
    #     type='TimerCP',
    # )
]
# load_from = None
# resume_from = None
