# Copyright (c) 2023-2024, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# TODO: add license here

import pickle

import numpy as np
from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

from tools.data_converter import nuscenes_converter as nuscenes_converter
# from tools.data_converter.nuscenes_prediction_tools import  get_forecasting_annotations
map_name_from_general_to_detection = {
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'ignore',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'animal': 'ignore',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.debris': 'ignore',
    'static_object.bicycle_rack': 'ignore',
}
classes = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]


VERSION= 'v1.0-mini'
NUSCENES = 'nuscenes-mini'
# VERSION= 'v1.0-trainval'
# NUSCENES = 'nuscenes'
def get_gt(info, traj_in_lidar_coor=None, traj_mask_in_lidar_coor=None):
    """Generate gt labels from info.

    Args:
        info(dict): Infos needed to generate gt labels.

    Returns:
        Tensor: GT bboxes.
        Tensor: GT labels.
    """

    ego2global_rotation = info['cams']['CAM_FRONT']['ego2global_rotation']
    ego2global_translation = info['cams']['CAM_FRONT'][
        'ego2global_translation']
    trans = -np.array(ego2global_translation)
    rot = Quaternion(ego2global_rotation).inverse
    gt_boxes = list()
    gt_boxes_in_global = list()
    gt_labels = list()
    fut_traj = list()
    fut_traj_mask = list()
    valid_flag = list()
    for i, ann_info in enumerate(info['ann_infos']):
        # Use ego coordinate.
        if (map_name_from_general_to_detection[ann_info['category_name']]
                not in classes
                or ann_info['num_lidar_pts'] + ann_info['num_radar_pts'] <= 0):
            valid_flag.append(False)
            continue
        valid_flag.append(True)
        box = Box(
            ann_info['translation'],
            ann_info['size'],
            Quaternion(ann_info['rotation']),
            velocity=ann_info['velocity'],
        )
        box_xyz_in_global = np.array(box.center)
        box_dxdydz_in_global = np.array(box.wlh)[[1, 0, 2]]
        box_yaw_in_global = np.array([box.orientation.yaw_pitch_roll[0]])
        box_velo_in_global = np.array(box.velocity[:2])

        box.translate(trans)
        box.rotate(rot)
        box_xyz = np.array(box.center)
        box_dxdydz = np.array(box.wlh)[[1, 0, 2]]
        box_yaw = np.array([box.orientation.yaw_pitch_roll[0]])
        box_velo = np.array(box.velocity[:2])
        gt_box = np.concatenate([box_xyz, box_dxdydz, box_yaw, box_velo])
        gt_box_in_global = np.concatenate([box_xyz_in_global, box_dxdydz_in_global, box_yaw_in_global, box_velo_in_global])
        gt_boxes.append(gt_box)
        gt_boxes_in_global.append(gt_box_in_global)
        gt_labels.append(
            classes.index(
                map_name_from_general_to_detection[ann_info['category_name']]))


        if traj_in_lidar_coor is not None:
            # traj = np.dot(Quaternion(info['lidar2ego_rotation']).rotation_matrix[:2,:2],traj_in_lidar_coor[i].transpose(1,0)).transpose(1,0)
            fut_traj.append(traj_in_lidar_coor[i])
            fut_traj_mask.append(traj_mask_in_lidar_coor[i])


    return gt_boxes, gt_labels, fut_traj, fut_traj_mask, np.array(valid_flag), gt_boxes_in_global

def nuscenes_data_prep(root_path, info_prefix, version, max_sweeps=10):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        max_sweeps (int, optional): Number of input consecutive frames.
            Default: 10
    """
    nuscenes_converter.create_nuscenes_infos(
        root_path, info_prefix, version=version, max_sweeps=max_sweeps)



def add_ann_adj_info(extra_tag, with_lidar_seg=False):
    nuscenes_version = VERSION
    dataroot = f'./data/{NUSCENES}/'
    nuscenes = NuScenes(nuscenes_version, dataroot)
    # for set in ['test']:
    #     dataset = pickle.load(
    #         open('./data/%s/%s_infos_%s.pkl' % (NUSCENES, extra_tag, set), 'rb'))
    #     for id in range(len(dataset['infos'])):
    #         if id % 10 == 0:
    #             print('%d/%d' % (id, len(dataset['infos'])))
    #         info = dataset['infos'][id]
    #         # get sweep adjacent frame info
    #         sample = nuscenes.get('sample', info['token'])
    #         ann_infos = list()
    #         for ann in sample['anns']:
    #             ann_info = nuscenes.get('sample_annotation', ann)
    #             velocity = nuscenes.box_velocity(ann_info['token'])
    #             if np.any(np.isnan(velocity)):
    #                 velocity = np.zeros(3)
    #             ann_info['velocity'] = velocity
    #             ann_infos.append(ann_info)
    #         dataset['infos'][id]['ann_infos'] = ann_infos
    #         dataset['infos'][id]['ann_infos'] = get_gt(dataset['infos'][id])
    #         dataset['infos'][id]['scene_token'] = sample['scene_token']
    #         scene = nuscenes.get('scene',  sample['scene_token'])
    #         dataset['infos'][id]['scene_name'] = scene['name']
    #         dataset['infos'][id]['prev'] = sample['prev']
    #         # description = scene['description']
    #         if with_lidar_seg:
    #             lidar_sd_token = sample['data']['LIDAR_TOP']
    #             dataset['infos'][id]['lidarseg_filename'] =  nuscenes.get('lidarseg', lidar_sd_token)['filename']


    #         scene = nuscenes.get('scene', sample['scene_token'])
    #         dataset['infos'][id]['occ_path'] = \
    #             './data/nuscenes/gts/%s/%s'%(scene['name'], info['token'])
    #     with open('./data/%s/%s_infos_%s.pkl' % (NUSCENES, extra_tag, set),
    #               'wb') as fid:
    #         pickle.dump(dataset, fid)

    for set in ['train', 'val']:
        dataset = pickle.load(
            open('./data/%s/%s_infos_%s.pkl' % (NUSCENES, extra_tag, set), 'rb'))
        # traj_data =  pickle.load(open(f'/mount/data/GoGo/data/infos/nuscenes_infos_temporal_{set}.pkl', 'rb'))            
        # traj_data = None
        for id in range(len(dataset['infos'])):
            if id % 10 == 0:
                print('%d/%d' % (id, len(dataset['infos'])))
            info = dataset['infos'][id]
            # get sweep adjacent frame info
            sample = nuscenes.get('sample', info['token'])
            ann_infos = list()
            for ann in sample['anns']:
                ann_info = nuscenes.get('sample_annotation', ann)
                velocity = nuscenes.box_velocity(ann_info['token'])
                if np.any(np.isnan(velocity)):
                    velocity = np.zeros(3)
                ann_info['velocity'] = velocity
                ann_infos.append(ann_info)
            dataset['infos'][id]['ann_infos'] = ann_infos
            # traj_info = traj_data['infos'][id] if traj_data is not None else None
            future_traj_all, future_traj_valid_mask_all = dataset['infos'][id]['fut_traj'],  dataset['infos'][id]['fut_traj_valid_mask']
            gt_boxes_3d, gt_labels_3d, fut_traj, fut_traj_mask, valid_flag, gt_boxes_3d_in_global = get_gt(dataset['infos'][id], future_traj_all, future_traj_valid_mask_all)

            dataset['infos'][id]['ann_infos'] = {}
            if fut_traj is not None:
                dataset['infos'][id]['ann_infos']['fut_traj'] = fut_traj
                dataset['infos'][id]['ann_infos']['fut_traj_mask'] = fut_traj_mask
            dataset['infos'][id]['ann_infos']['gt_boxes_2d'] = dataset['infos'][id]['bboxes2d']
            dataset['infos'][id]['ann_infos']['gt_labels_2d'] = dataset['infos'][id]['labels2d']
            dataset['infos'][id]['ann_infos']['depths'] = dataset['infos'][id]['depths']            
            dataset['infos'][id]['ann_infos']['centers2d'] = dataset['infos'][id]['centers2d']

            dataset['infos'][id]['ann_infos']['gt_boxes_3d'] = gt_boxes_3d
            dataset['infos'][id]['ann_infos']['gt_boxes_3d_in_global'] = gt_boxes_3d_in_global
            dataset['infos'][id]['ann_infos']['gt_labels_3d'] = gt_labels_3d
            dataset['infos'][id]['scene_token'] = sample['scene_token']
            scene = nuscenes.get('scene',  sample['scene_token'])
            map_location = nuscenes.get('log', scene['log_token'])['location']
            dataset['infos'][id]['map_location'] = map_location
            dataset['infos'][id]['scene_name'] = scene['name']
            dataset['infos'][id]['prev'] = sample['prev']
            
            annotations = [
                nuscenes.get('sample_annotation', token)
                for token in sample['anns']
            ]

            instance_inds = [nuscenes.getind('instance', ann['instance_token']) for ann in annotations]
            info['instance_inds'] = instance_inds
            info['valid_flag'] = valid_flag

            # description = scene['description']
            if with_lidar_seg:
                lidar_sd_token = sample['data']['LIDAR_TOP']
                dataset['infos'][id]['lidarseg_filename'] =  nuscenes.get('lidarseg', lidar_sd_token)['filename']
            scene = nuscenes.get('scene', sample['scene_token'])
            dataset['infos'][id]['occ_path'] = \
                './data/nuscenes/gts/%s/%s'%(scene['name'], info['token'])
        with open('./data/%s/%s_infos_%s.pkl' % (NUSCENES, extra_tag, set),
                  'wb') as fid:
            pickle.dump(dataset, fid)


if __name__ == '__main__':
    dataset = 'nuscenes'
    version = 'v1.0'
    train_version = VERSION
    root_path = f'./data/{NUSCENES}'
    extra_tag = 'bev-next-nuscenes'
    nuscenes_data_prep(
        root_path=root_path,
        info_prefix=extra_tag,
        version=train_version,
        max_sweeps=10)

    print('add_ann_infos')
    add_ann_adj_info(extra_tag)
