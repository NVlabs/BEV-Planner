# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/FB-BEV/blob/main/LICENSE


# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
import copy
from os import path as osp
import os
import mmcv
import sys
import numpy as np
import pyquaternion
from nuscenes.utils.data_classes import Box as NuScenesBox
from .utils import nuscenes_get_rt_matrix
from ..core import show_result
from ..core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes
from .builder import DATASETS
from .custom_3d import Custom3DDataset
from .pipelines import Compose
from tqdm import tqdm
import csv
import math
import torch
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
# from .vad_custom_nuscenes_eval import NuScenesEval_custom
from nuscenes.eval.common.utils import center_distance
# from projects.mmdet3d_plugin.models.utils.visual import save_tensor
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
from .vector_map import VectorizedLocalMap

@DATASETS.register_module()
class NuScenesDataset(Custom3DDataset):
    r"""NuScenes Dataset.

    This class serves as the API for experiments on the NuScenes Dataset.

    Please refer to `NuScenes Dataset <https://www.nuscenes.org/download>`_
    for data downloading.

    Args:
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        data_root (str): Path of dataset root.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        load_interval (int, optional): Interval of loading the dataset. It is
            used to uniformly sample the dataset. Defaults to 1.
        with_velocity (bool, optional): Whether include velocity prediction
            into the experiments. Defaults to True.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes.
            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        eval_version (bool, optional): Configuration version of evaluation.
            Defaults to  'detection_cvpr_2019'.
        use_valid_flag (bool, optional): Whether to use `use_valid_flag` key
            in the info file as mask to filter gt_boxes and gt_names.
            Defaults to False.
        img_info_prototype (str, optional): Type of img information.
            Based on 'img_info_prototype', the dataset will prepare the image
            data info in the type of 'mmcv' for official image infos,
            'bevdet' for BEVDet, and 'bevdet4d' for BEVDet4D.
            Defaults to 'mmcv'.
        multi_adj_frame_id_cfg (tuple[int]): Define the selected index of
            reference adjcacent frames.
        ego_cam (str): Specify the ego coordinate relative to a specified
            camera by its name defined in NuScenes.
            Defaults to None, which use the mean of all cameras.
    """
    NameMapping = {
        'movable_object.barrier': 'barrier',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.motorcycle': 'motorcycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'movable_object.trafficcone': 'traffic_cone',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck'
    }
    DefaultAttribute = {
        'car': 'vehicle.parked',
        'pedestrian': 'pedestrian.moving',
        'trailer': 'vehicle.parked',
        'truck': 'vehicle.parked',
        'bus': 'vehicle.moving',
        'motorcycle': 'cycle.without_rider',
        'construction_vehicle': 'vehicle.parked',
        'bicycle': 'cycle.without_rider',
        'barrier': '',
        'traffic_cone': '',
    }
    AttrMapping = {
        'cycle.with_rider': 0,
        'cycle.without_rider': 1,
        'pedestrian.moving': 2,
        'pedestrian.standing': 3,
        'pedestrian.sitting_lying_down': 4,
        'vehicle.moving': 5,
        'vehicle.parked': 6,
        'vehicle.stopped': 7,
    }
    AttrMapping_rev = [
        'cycle.with_rider',
        'cycle.without_rider',
        'pedestrian.moving',
        'pedestrian.standing',
        'pedestrian.sitting_lying_down',
        'vehicle.moving',
        'vehicle.parked',
        'vehicle.stopped',
    ]
    # https://github.com/nutonomy/nuscenes-devkit/blob/57889ff20678577025326cfc24e57424a829be0a/python-sdk/nuscenes/eval/detection/evaluate.py#L222 # noqa
    ErrNameMapping = {
        'trans_err': 'mATE',
        'scale_err': 'mASE',
        'orient_err': 'mAOE',
        'vel_err': 'mAVE',
        'attr_err': 'mAAE'
    }
    CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
               'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
               'barrier')
    
    TRACKING_CLASSES = ['car', 'truck', 'bus', 'trailer',
               'motorcycle', 'bicycle', 'pedestrian']

    def __init__(self,
                 ann_file=None,
                 pipeline=None,
                 data_root=None,
                 classes=None,
                 load_interval=1,
                 with_velocity=True,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 eval_version='detection_cvpr_2019',
                 use_valid_flag=False,
                 img_info_prototype='mmcv',
                 multi_adj_frame_id_cfg=None,
                 occupancy_path='/mount/dnn_data/occupancy_2023/gts',
                 ego_cam='CAM_FRONT',
                 # SOLLOFusion
                 use_sequence_group_flag=False,
                 sequences_split_num=1,
                 # MAP
                 map_classes = ['divider', 'ped_crossing', 'boundary'],
                 map_ann_file= '',
                 point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 map_eval_cfg=dict(),
                 load_fut_bbox_info=False,
                ):
        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag

        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode)
        self.load_fut_bbox_info = load_fut_bbox_info
        self.occupancy_path = occupancy_path
        self.with_velocity = with_velocity
        self.eval_version = eval_version
        from nuscenes.eval.detection.config import config_factory

        self.eval_detection_configs = config_factory(self.eval_version)
        if self.modality is None:
            self.modality = dict(
                use_camera=False,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )
        self.map_eval_cfg = map_eval_cfg
        self.map_ann_file = map_ann_file
        self.MAPCLASSES = self.get_map_classes(map_classes)
        self.NUM_MAPCLASSES = len(self.MAPCLASSES)
        self.pc_range = point_cloud_range

        self.img_info_prototype = img_info_prototype
        self.multi_adj_frame_id_cfg = multi_adj_frame_id_cfg
        self.ego_cam = ego_cam
        self.nusc = None

        # SOLOFusion
        self.use_sequence_group_flag = use_sequence_group_flag
        self.sequences_split_num = sequences_split_num
        # sequences_split_num splits eacgh sequence into sequences_split_num parts.
        # if self.test_mode:
        #     assert self.sequences_split_num == 1
        if self.use_sequence_group_flag:
            self._set_sequence_group_flag() # Must be called after load_annotations b/c load_annotations does sorting.



    def get_cat_ids(self, idx):
        """Get category distribution of single scene.

        Args:
            idx (int): Index of the data_info.

        Returns:
            dict[list]: for each category, if the current scene
                contains such boxes, store a list containing idx,
                otherwise, store empty list.
        """
        info = self.data_infos[idx]
        if self.use_valid_flag:
            mask = info['valid_flag']
            gt_names = set(info['gt_names'][mask])
        else:
            gt_names = set(info['gt_names'])

        cat_ids = []
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file, file_format='pkl')

        data_infos = data['infos'][::self.load_interval]
        self.metadata = data['metadata']

        self.version = self.metadata['version']
        if len(data_infos) < 100:
            self.version = 'v1.0-mini'
        return data_infos


    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group
        """
           
        res = []
        curr_sequence = 0
        for idx in range(len(self.data_infos)):
            if idx != 0 and len(self.data_infos[idx]['prev']) == 0:
                # Not first frame and # of sweeps is 0 -> new sequence
                curr_sequence += 1
            res.append(curr_sequence)
        self.flag = np.array(res, dtype=np.int64)
        if self.sequences_split_num != 1:
            if self.sequences_split_num == 'all':
                self.flag = np.array(range(len(self.data_infos)), dtype=np.int64)
            else:
                bin_counts = np.bincount(self.flag)
                new_flags = []
                curr_new_flag = 0
                for curr_flag in range(len(bin_counts)):
                    curr_sequence_length = np.array(
                        list(range(0, 
                                bin_counts[curr_flag], 
                                math.ceil(bin_counts[curr_flag] / self.sequences_split_num)))
                        + [bin_counts[curr_flag]])
                    for sub_seq_idx in (curr_sequence_length[1:] - curr_sequence_length[:-1]):
                        for _ in range(sub_seq_idx):
                            new_flags.append(curr_new_flag)
                        curr_new_flag += 1

                assert len(new_flags) == len(self.flag)
                assert len(np.bincount(new_flags)) == len(np.bincount(self.flag)) * self.sequences_split_num
                self.flag = np.array(new_flags, dtype=np.int64)

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = copy.deepcopy(self.data_infos[index])
        # standard protocol modified from SECOND.Pytorch
        input_dict = dict(
            index=index,
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            scene_name=info['scene_name'],
            timestamp=info['timestamp'] / 1e6,
            lidarseg_filename=info.get('lidarseg_filename', 'None') 
        )

        if 'instance_inds' in info.keys():
            assert len(info['instance_inds']) == len(info['valid_flag'])
            if len(info['instance_inds'])>0:
                input_dict['instance_inds'] = np.array(info['instance_inds'])[info['valid_flag']]
            else: input_dict['instance_inds'] = np.array(info['instance_inds'])

        if 'ann_infos' in info:
            input_dict['ann_infos'] = info['ann_infos']
            
        if self.modality['use_camera']:
            if self.img_info_prototype == 'mmcv':
                image_paths = []
                lidar2img_rts = []

                for cam_type, cam_info in info['cams'].items():
                    image_paths.append(cam_info['data_path'])
                    # obtain lidar to image transformation matrix
                    lidar2cam_r = np.linalg.inv(
                        cam_info['sensor2lidar_rotation'])
                    lidar2cam_t = cam_info[
                        'sensor2lidar_translation'] @ lidar2cam_r.T
                    lidar2cam_rt = np.eye(4)
                    lidar2cam_rt[:3, :3] = lidar2cam_r.T
                    lidar2cam_rt[3, :3] = -lidar2cam_t
                    intrinsic = cam_info['cam_intrinsic']
                    viewpad = np.eye(4)
                    viewpad[:intrinsic.shape[0], :intrinsic.
                            shape[1]] = intrinsic
                    lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                    lidar2img_rts.append(lidar2img_rt)
                    cam_position = np.linalg.inv(lidar2cam_rt.T) @ np.array([0., 0., 0., 1.]).reshape([4, 1])
                    cam_positions.append(cam_position.flatten()[:3])
                   

                input_dict.update(
                    dict(
                        
                        img_filename=image_paths,
                        lidar2img=lidar2img_rts,
                    ))

                if not self.test_mode:
                    annos = self.get_ann_info(index)
                    input_dict['ann_info'] = annos
            else:   
                assert 'bevdet' in self.img_info_prototype
                input_dict.update(dict(curr=info))
                if '4d' in self.img_info_prototype:
                    info_adj_list = self.get_adj_info(info, index)
                    input_dict.update(dict(adjacent=info_adj_list))
            if self.use_sequence_group_flag:
                input_dict['sample_index'] = index
                input_dict['sequence_group_idx'] = self.flag[index]
                input_dict['start_of_sequence'] = index == 0 or self.flag[index - 1] != self.flag[index]
                # Get a transformation matrix from current keyframe lidar to previous keyframe lidar
                # if they belong to same sequence.
                can_bus_info = info['gt_ego_lcf_feat']
                input_dict['can_bus_info'] = can_bus_info
                input_dict['nuscenes_get_rt_matrix'] = dict(
                    lidar2ego_rotation = info['lidar2ego_rotation'],
                    lidar2ego_translation = info['lidar2ego_translation'],
                    ego2global_rotation = info['ego2global_rotation'],
                    ego2global_translation = info['ego2global_translation'],
                )

                input_dict['ego_pose_inv'] = torch.FloatTensor(nuscenes_get_rt_matrix(
                    info, info,
                    "global", "ego"))
                
                input_dict['ego_pose'] = torch.FloatTensor(nuscenes_get_rt_matrix(
                    info, info,
                    "ego", "global"))
                
                

                if not input_dict['start_of_sequence']:
                    input_dict['curr_to_prev_lidar_rt'] = torch.FloatTensor(nuscenes_get_rt_matrix(
                        info, self.data_infos[index - 1],
                        "lidar", "lidar"))
                    input_dict['prev_lidar_to_global_rt'] = torch.FloatTensor(nuscenes_get_rt_matrix(
                        self.data_infos[index - 1], info,
                        "lidar", "global")) # TODO: Note that global is same for all.
                    input_dict['curr_to_prev_ego_rt'] = torch.FloatTensor(nuscenes_get_rt_matrix(
                       info, self.data_infos[index - 1],
                        "ego", "ego"))
                else:
                    input_dict['curr_to_prev_lidar_rt'] = torch.eye(4).float()
                    input_dict['prev_lidar_to_global_rt'] = torch.FloatTensor(nuscenes_get_rt_matrix( 
                        info, info, "lidar", "global")
                        )
                    input_dict['curr_to_prev_ego_rt'] = torch.FloatTensor(nuscenes_get_rt_matrix(
                        info, info,
                        "ego", "ego"))
                input_dict['global_to_curr_lidar_rt'] = torch.FloatTensor(nuscenes_get_rt_matrix(
                    info, info,
                    "global", "lidar"))
                

                if self.load_fut_bbox_info:
                    fut_boxes_info, fut_labels_info = self.get_fut_bbox_info(info, index)
                    input_dict['fut_boxes_info'] = fut_boxes_info
                    input_dict['fut_labels_info'] = fut_labels_info
                 
        return input_dict

    def get_fut_bbox_info(self, info, index):
        fut_boxes_info = []
        fut_labels_info = []
        for select_id in range(1, 7):
            select_id = min(index + select_id, len(self.data_infos)-1)
            if not self.data_infos[select_id]['scene_token'] == info[
                    'scene_token']:
                fut_boxes_info.append([])
                fut_labels_info.append([])
            else:
                fut_boxes_info.append(self.data_infos[select_id]['ann_infos']['gt_boxes_3d_in_global'])
                fut_labels_info.append(self.data_infos[select_id]['ann_infos']['gt_labels_3d'])

        return fut_boxes_info, fut_labels_info


    def get_adj_info(self, info, index):
        info_adj_list = []
        for select_id in range(*self.multi_adj_frame_id_cfg):
            if select_id == 0: continue
            select_id = min(max(index - select_id, 0), len(self.data_infos)-1)

            if not self.data_infos[select_id]['scene_token'] == info[
                    'scene_token']:
                info_adj_list.append(info)
            else:
                info_adj_list.append(self.data_infos[select_id])
        return info_adj_list

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info['valid_flag']
        else:
            mask = info['num_lidar_pts'] > 0
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info['gt_velocity'][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d)
        return anns_results


    def format_map_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a \
                dict containing the json filepaths, `tmp_dir` is the temporal \
                directory created for saving json files when \
                `jsonfile_prefix` is not specified.
        """
        if isinstance(results, dict):
            results = results['map_results']
        assert isinstance(results, list)
        assert len(results) >= len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        # currently the output prediction results could be in two formats
        # 1. list of dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...)
        # 2. list of dict('pts_bbox' or 'img_bbox':
        #     dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...))
        # this is a workaround to enable evaluation of both formats on nuScenes
        # refer to https://github.com/open-mmlab/mmdetection3d/issues/449
        if not ('pred_map' in results[0]):
            result_files = self._format_map(results, jsonfile_prefix)
        else:
            # should take the inner dict out of 'pts_bbox' or 'img_bbox' dict
            result_files = dict()
            for name in ['pred_map']:
                print(f'\nFormating {name}')
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_files.update(
                    {name: self._format_map(results_, tmp_file_)})
        return result_files, tmp_dir

    @classmethod
    def get_map_classes(cls, map_classes=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.

        Return:
            list[str]: A list of class names.
        """
        if map_classes is None:
            return cls.MAPCLASSES

        if isinstance(map_classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(map_classes)
        elif isinstance(map_classes, (tuple, list)):
            class_names = map_classes
        else:
            raise ValueError(f'Unsupported type {type(map_classes)} of map classes.')

        return class_names

    def _format_map(self, results, jsonfile_prefix=None, score_thresh=0.2):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """

        # assert self.map_ann_file is not None
        map_pred_annos = {}
        map_mapped_class_names = self.MAPCLASSES
        processed_set = set()
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            sample_id = det.get('index', sample_id)
            if sample_id in processed_set: continue
            processed_set.add(sample_id)
            map_pred_anno = {}
            vecs = output_to_vecs(det)
            sample_token = self.data_infos[sample_id]['token']
            map_pred_anno['sample_token'] = sample_token
            pred_vec_list=[]
            for i, vec in enumerate(vecs):
                name = map_mapped_class_names[vec['label']]
                anno = dict(
                    sample_token=sample_token,
                    pts=vec['pts'],
                    pts_num=len(vec['pts']),
                    cls_name=name,
                    type=vec['label'],
                    confidence_level=vec['score'])
                pred_vec_list.append(anno)
                # annos.append(nusc_anno)
            # nusc_annos[sample_token] = annos
            map_pred_anno['vectors'] = pred_vec_list
            map_pred_annos[sample_token] = map_pred_anno
        # self._format_map_gt()

        if not os.path.exists(self.map_ann_file):
            self._format_map_gt()
        else:
            print(f'{self.map_ann_file} exist, not update')
        # with open(self.map_ann_file,'r') as f:
        #     GT_anns = json.load(f)
        # gt_annos = GT_anns['GTs']

        nusc_submissions = {
            'meta': self.modality,
            'map_results': map_pred_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'map_results_nusc.json')
        print('Map Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    def vectormap_pipeline(self, location, ego2global_translation, patch_angle, flip_dx=False, flip_dy=False):
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
        
        '''
        anns_results, type: dict
            'gt_vecs_pts_loc': list[num_vecs], vec with num_points*2 coordinates
            'gt_vecs_pts_num': list[num_vecs], vec with num_points
            'gt_vecs_label': list[num_vecs], vec with cls index
        '''
        gt_vecs_label = to_tensor(anns_results['gt_vecs_label'])
        from .vector_map import LiDARInstanceLines
        if isinstance(anns_results['gt_vecs_pts_loc'], LiDARInstanceLines):
            gt_vecs_pts_loc = anns_results['gt_vecs_pts_loc']
        else:
            gt_vecs_pts_loc = to_tensor(anns_results['gt_vecs_pts_loc'])
            try:
                gt_vecs_pts_loc = gt_vecs_pts_loc.flatten(1).to(dtype=torch.float32)
            except:
                # empty tensor, will be passed in train, 
                # but we preserve it for test
                gt_vecs_pts_loc = gt_vecs_pts_loc

        return dict(
            map_gt_labels_3d = DC(gt_vecs_label, cpu_only=False),
            map_gt_bboxes_3d = DC(gt_vecs_pts_loc, cpu_only=True),
        )

    def _format_map_gt(self):
        gt_annos = []
        print('Start to convert gt map format...')
        # assert self.map_ann_file is not None

        if  (not os.path.exists(self.map_ann_file)) :

            patch_h, patch_w = self.map_eval_cfg['region']
            patch_h = min(patch_h, 50)
            self.vector_map = VectorizedLocalMap(self.data_root, 
                            patch_size=(patch_h, patch_w), map_classes=self.MAPCLASSES, 
                            fixed_ptsnum_per_line=20,
                            padding_value=-10000)

            dataset_length = len(self)
            prog_bar = mmcv.ProgressBar(dataset_length)
            mapped_class_names = self.MAPCLASSES
            for sample_id in range(dataset_length):
                sample_token = self.data_infos[sample_id]['token']
                gt_anno = {}
                gt_anno['sample_token'] = sample_token
                # gt_sample_annos = []
                gt_sample_dict = {}
                
                ego_pose = torch.FloatTensor(nuscenes_get_rt_matrix(
                    self.data_infos[sample_id], self.data_infos[sample_id],
                    "ego", "global"))

                ego2global_translation = list(ego_pose[:3,3].numpy())
                v = np.dot(ego_pose[:3,:3].numpy(), np.array([1, 0, 0]))
                yaw = np.arctan2(v[1], v[0])
                patch_angle = yaw / np.pi * 180
                location = self.data_infos[sample_id]['map_location']

                gt_sample_dict =  self.vectormap_pipeline(location, ego2global_translation, patch_angle)
                gt_labels = gt_sample_dict['map_gt_labels_3d'].data.numpy()
                gt_vecs = gt_sample_dict['map_gt_bboxes_3d'].data.instance_list
                gt_vec_list = []
                for i, (gt_label, gt_vec) in enumerate(zip(gt_labels, gt_vecs)):
                    name = mapped_class_names[gt_label]
                    anno = dict(
                        pts=np.array(list(gt_vec.coords)),
                        pts_num=len(list(gt_vec.coords)),
                        cls_name=name,
                        type=gt_label,
                    )
                    gt_vec_list.append(anno)
                gt_anno['vectors']=gt_vec_list
                gt_annos.append(gt_anno)

                prog_bar.update()
            nusc_submissions = {
                'GTs': gt_annos
            }
            print('\n GT anns writes to', self.map_ann_file)
            mmcv.dump(nusc_submissions, self.map_ann_file)
        else:
            print(f'{self.map_ann_file} exist, not update')


    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            metric (str, optional): Metric name used for evaluation.
                Default: 'bbox'.
            result_name (str, optional): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        from nuscenes import NuScenes
        from nuscenes.eval.detection.evaluate import NuScenesEval

        output_dir = osp.join(*osp.split(result_path)[:-1])

        self.nusc = NuScenes(
            version=self.version, dataroot=self.data_root, verbose=False)
        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        nusc_eval = NuScenesEval(
            self.nusc,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=False)
        nusc_eval.main(render_curves=False)

        # record metrics
        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = f'{result_name}_NuScenes'
        for name in self.CLASSES:
            for k, v in metrics['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix,
                                      self.ErrNameMapping[k])] = val

        detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
        detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']
        return detail

    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a
                dict containing the json filepaths, `tmp_dir` is the temporal
                directory created for saving json files when
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) >= len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        # currently the output prediction results could be in two formats
        # 1. list of dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...)
        # 2. list of dict('pts_bbox' or 'img_bbox':
        #     dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...))
        # this is a workaround to enable evaluation of both formats on nuScenes
        # refer to https://github.com/open-mmlab/mmdetection3d/issues/449
        if not ('pts_bbox' in results[0] or 'img_bbox' in results[0]):
            result_files = self._format_bbox(results, jsonfile_prefix)
        else:
            # should take the inner dict out of 'pts_bbox' or 'img_bbox' dict
            result_files = dict()
            for name in ['pts_bbox']:
                print(f'\nFormating bboxes of {name}')
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_files.update(
                    {name: self._format_bbox(results_, tmp_file_)})
        return result_files, tmp_dir

    # def format_motion_results(self, results, jsonfile_prefix=None):
    #     """Format the results to json (standard format for COCO evaluation).

    #     Args:
    #         results (list[dict]): Testing results of the dataset.
    #         jsonfile_prefix (str): The prefix of json files. It includes
    #             the file path and the prefix of filename, e.g., "a/b/prefix".
    #             If not specified, a temp file will be created. Default: None.

    #     Returns:
    #         tuple: Returns (result_files, tmp_dir), where `result_files` is a
    #             dict containing the json filepaths, `tmp_dir` is the temporal
    #             directory created for saving json files when
    #             `jsonfile_prefix` is not specified.
    #     """
    #     assert isinstance(results, list), 'results must be a list'
    #     assert len(results) >= len(self), (
    #         'The length of results is not equal to the dataset len: {} != {}'.
    #         format(len(results), len(self)))

    #     if jsonfile_prefix is None:
    #         tmp_dir = tempfile.TemporaryDirectory()
    #         jsonfile_prefix = osp.join(tmp_dir.name, 'results')
    #     else:
    #         tmp_dir = None

    #     # currently the output prediction results could be in two formats
    #     # 1. list of dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...)
    #     # 2. list of dict('pts_bbox' or 'img_bbox':
    #     #     dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...))
    #     # this is a workaround to enable evaluation of both formats on nuScenes
    #     # refer to https://github.com/open-mmlab/mmdetection3d/issues/449
    #     if not ('pred_motion' in results[0]):
    #         result_files = self._format_motion_bbox(results, jsonfile_prefix)
    #     else:
    #         # should take the inner dict out of 'pts_bbox' or 'img_bbox' dict
    #         result_files = dict()
    #         for name in ['pred_motion']:
    #             print(f'\nFormating bboxes of {name}')
    #             results_ = [out[name] for out in results]
    #             tmp_file_ = osp.join(jsonfile_prefix, name)
    #             result_files.update(
    #                 {name: self._format_motion_bbox(results_, tmp_file_)})
    #     return result_files, tmp_dir

    def evaluate(self, results,
                       logger=None,
                        metric='bbox',
                        jsonfile_prefix='test',
                        result_names=['pts_bbox'],
                        show=False,
                        out_dir=None,
                        pipeline=None,
                        save=False,
                        ):
            results_dict = {}
            mmcv.mkdir_or_exist(jsonfile_prefix)

            if results[0].get('pred_ego_traj', None) is not None:
                results_dict.update(
                    self.evaluate_ego_traj(
                        results,
                        jsonfile_prefix=jsonfile_prefix,
                        logger=logger
                    )
                )
            if results[0].get('pred_occupancy', None) is not None:
                results_dict.update(self.evaluate_occupancy(results, show_dir=jsonfile_prefix, save=save))
                
            if results[0].get('iou', None) is not None:
                results_dict.update(self.evaluate_mask(results))
            
            if results[0].get('pred_map', None) is not None:
                results_dict.update(self.evaluate_map(results, jsonfile_prefix=jsonfile_prefix, out_dir=out_dir))


            if results[0].get('pts_bbox', None) is not None:

                results_dict.update(self.evaluate_bbox(results, logger=logger,
                        metric=metric,
                        jsonfile_prefix=jsonfile_prefix,
                        result_names=result_names,
                        show=show,
                        out_dir=out_dir,
                        pipeline=pipeline))
                
                """if the output information has no tracking info, this func dose nothing"""
                results_dict.update(self.evaluate_tracking(results, logger=logger,
                        metric=metric,
                        jsonfile_prefix=jsonfile_prefix,
                        result_names=result_names,
                        show=show,
                        out_dir=out_dir,
                        pipeline=pipeline))

            with open(osp.join(jsonfile_prefix, 'results.csv'), 'w', newline='') as f:
                writer = csv.writer(f)
                for key in results_dict.keys():
                    writer.writerow([key, results_dict[key]])

            return results_dict



    def evaluate_ego_traj(self, results, jsonfile_prefix=None, logger=None):
        print('Start to convert traj format...')
        l2_dist_list = []
        res = torch.zeros(1, 6)
        res_c = torch.zeros(1, 6)
        processed_set = set()
        ego_trajs_in_global_dict = dict(
            trajs=dict(),
            map_lane=dict(),
            map_label=dict(),
        )
        c = 0
        
        gen_global_map = False
        if gen_global_map:
            self.vector_map = VectorizedLocalMap(self.data_root, 
                            patch_size=(400, 400), map_classes=self.MAPCLASSES, 
                            fixed_ptsnum_per_line=200,
                            padding_value=-10000)
        
        for sample_id, traj in enumerate(mmcv.track_iter_progress(results)):
            sample_id = traj['pred_ego_traj']['index']
            l2_dist = traj['pred_ego_traj']['metric_dict'].pop('l2_dist')
            if sample_id in processed_set: continue
            # if traj['pred_ego_traj']['gt_ego_fut_cmd'][-1] == 1: continue
            processed_set.add(sample_id)
            c += 1
            ego_trajs_in_global = traj['pred_ego_traj']['ego_trajs_in_global'].numpy()
            ego_trajs_in_global_dict['trajs'][traj['pred_ego_traj']['index_w_scene']] = ego_trajs_in_global
            mask = l2_dist >= 0
            res[mask] += l2_dist[mask]
            res_c[mask] += 1
            info = self.data_infos[sample_id]
            # print(traj['pred_ego_traj']['index_w_scene'], info['prev']=='', sample_id, traj['pred_ego_traj']['index'])
            if gen_global_map and info['prev']=='':
                
                ego_pose = torch.FloatTensor(nuscenes_get_rt_matrix(
                        self.data_infos[sample_id], self.data_infos[sample_id],
                        "ego", "global"))
                ego2global_translation = list(ego_pose[:3,3].numpy())
                map_res = self.vectormap_pipeline(info['map_location'], ego2global_translation, 0)
                lanes = map_res['map_gt_bboxes_3d'].data.fixed_num_sampled_points.cpu().numpy() 
                lanes = lanes + ego2global_translation[:2]
                lanes_label = map_res['map_gt_labels_3d'].data.cpu().numpy()
                ego_trajs_in_global_dict['map_lane'][traj['pred_ego_traj']['index_w_scene']] = lanes
                # results[sample_id]['pred_map']['gt_lane_in_global']
                ego_trajs_in_global_dict['map_label'][traj['pred_ego_traj']['index_w_scene']] = lanes_label
                #     results[sample_id]['pred_map']['gt_lane_label']

        print('valid: ', c)

        l2_dist = (res/res_c).cpu().numpy()
       
        print('++++++++++++++')
        print('l2_dist')
        print(l2_dist)
        print('--------------')


        metric_dict = [None, None, None]

        for i in range(3):
            num_valid = 0
            processed_set = set()
            for sample_id, traj in enumerate(mmcv.track_iter_progress(results)):
                sample_id = traj['pred_ego_traj']['index']
                if sample_id in processed_set: continue
                if i == 1 and traj['pred_ego_traj']['gt_ego_fut_cmd'][-1] == 1: continue
                if i == 2 and traj['pred_ego_traj']['gt_ego_fut_cmd'][-1] != 1: continue
                processed_set.add(sample_id)
                if not traj['pred_ego_traj']['metric_dict']['fut_valid_flag']: continue
                else: num_valid += 1

                if metric_dict[i] is None:
                    metric_dict[i] = copy.deepcopy(traj['pred_ego_traj']['metric_dict'])
                else:
                    for k in traj['pred_ego_traj']['metric_dict'].keys():
                        metric_dict[i][k] += traj['pred_ego_traj']['metric_dict'][k]

            print('valid_after: ', num_valid, i)
            for k in metric_dict[i]:
                metric_dict[i][k] = str(metric_dict[i][k] / num_valid)
                print("{}:{}:{}".format(i, k, metric_dict[i][k]))

        res_path = osp.join(jsonfile_prefix, 'results_nusc_planning.json')
        print('Results writes to', res_path)
    
        mmcv.dump(ego_trajs_in_global_dict, res_path)
        metric_dict[0].update(self.smoothness(ego_trajs_in_global_dict['trajs']))
        #     l2_dist_1s = traj['pred_ego_traj']['metric_dict']['plan_L2_1s']
        #     l2_dist_2s = traj['pred_ego_traj']['metric_dict']['plan_L2_2s']
        #     l2_dist_3s = traj['pred_ego_traj']['metric_dict']['plan_L2_3s']
        #     res2[0] = res2[0] + l2_dist_1s
        #     res2[1] = res2[1] + l2_dist_2s
        #     res2[2] = res2[2] + l2_dist_3s
        #     res2_c += 1
        # l2_dist_v2 = res2/res2_c
        
        # print('++++++++++++++')
        # print('l2_dist_v2')
        # print(l2_dist_v2)
        avg_l2 = 0
        avg_col = 0
        for i in range(1,4):
            avg_l2 += float(metric_dict[0][f'plan_L2_{i}s'])
            avg_col += float(metric_dict[0][f'plan_obj_box_col_{i}s'])
        avg_l2 /= 3
        avg_col /= 3
        print(f'avg_l2 {avg_l2}, avg_col {avg_col}')
        print('--------------')
        # metric_dict['l2_dist'] = l2_dist
        metric_dict[0]['avg_l2'] = avg_l2
        metric_dict[0]['avg_col'] = avg_col
        return metric_dict[0]

    def smoothness(self, data):
        keys = list(data.keys())
        # print(keys)
        new_keys = []
        for key in keys:
            s = key.split("-")
            new_keys.append([int(s[1]),int(s[2])])

        new_keys=sorted(new_keys,key=(lambda x:(x[0], x[1])))
        sorted_keys = []
        for key in new_keys:
            v = ['scene',  str(key[0]).zfill(4), str(key[1]) ]
            k='-'.join(v)
            sorted_keys.append(k)


        all_scene_keys=[]
        key='-'.join(sorted_keys[0].split("-")[:2])
        scene=[]

        for k in sorted_keys:
            if(key in k):
                # print(True)
                scene.append(k)
            else:
                s =k.split("-")
                key='-'.join(s[:2])
                all_scene_keys.append(scene)
                scene=[k]

        #tranform raw data
        new_data={}
        for keys in all_scene_keys:
            l = len(keys)
            for i in range(l):
                val = []
                index = i
                for j in range(i+1):
                    if index>6:
                        index-=1
                    else:
                        val.append(data[keys[j]][index])
                        index-=1
                new_data[keys[i]]=val

        #compute mean and var
        res = {
            'stable_mean_distance_1s': [],
            'stable_variance_distance_1s': [],
            'stable_mean_distance_2s': [],
            'stable_variance_distance_2s': [],
            'stable_mean_distance_3s': [],
            'stable_variance_distance_3s': [],
        }
        
        for key, value in new_data.items():
            #filter unstable data
            if(len(value)!=7):
                continue
            assert len(value)==7
            #compute mean
            for window in [1, 2, 3]:
                gt = value[-1]
                pred = value[6-window*2:-1]
                #compute var
                data_array = np.array(pred)

                distances = np.linalg.norm(data_array - gt, axis=1)
                mean_distance = np.mean(distances)
                variance_distance = np.var(distances)
                res[f'stable_mean_distance_{window}s'].append(mean_distance)
                res[f'stable_variance_distance_{window}s'].append(variance_distance)
        
        for key in res.keys():
            res[key] = np.mean(res[key])
        print(res)
        return res
    
    def _format_bbox(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}
        mapped_class_names = self.CLASSES
        print('Start to convert detection format...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            boxes = det['boxes_3d'].tensor.numpy()
            scores = det['scores_3d'].numpy()
            labels = det['labels_3d'].numpy()
            sample_id = det.get('index', sample_id)

            sample_token = self.data_infos[sample_id]['token']

            trans = self.data_infos[sample_id]['cams'][
                self.ego_cam]['ego2global_translation']
            rot = self.data_infos[sample_id]['cams'][
                self.ego_cam]['ego2global_rotation']
            rot = pyquaternion.Quaternion(rot)
            annos = list()
            for i, box in enumerate(boxes):
                name = mapped_class_names[labels[i]]
                center = box[:3]
                wlh = box[[4, 3, 5]]
                box_yaw = box[6]
                box_vel = box[7:].tolist()
                box_vel.append(0)
                quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw)
                nusc_box = NuScenesBox(center, wlh, quat, velocity=box_vel)
                nusc_box.rotate(rot)
                nusc_box.translate(trans)
                if np.sqrt(nusc_box.velocity[0]**2 +
                           nusc_box.velocity[1]**2) > 0.2:
                    if name in [
                            'car',
                            'construction_vehicle',
                            'bus',
                            'truck',
                            'trailer',
                    ]:
                        attr = 'vehicle.moving'
                    elif name in ['bicycle', 'motorcycle']:
                        attr = 'cycle.with_rider'
                    else:
                        attr = self.DefaultAttribute[name]
                else:
                    if name in ['pedestrian']:
                        attr = 'pedestrian.standing'
                    elif name in ['bus']:
                        attr = 'vehicle.stopped'
                    else:
                        attr = self.DefaultAttribute[name]
                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=nusc_box.center.tolist(),
                    size=nusc_box.wlh.tolist(),
                    rotation=nusc_box.orientation.elements.tolist(),
                    velocity=nusc_box.velocity[:2],
                    detection_name=name,
                    detection_score=float(scores[i]),
                    attribute_name=attr,
                )
                annos.append(nusc_anno)
            # other views results of the same frame should be concatenated
            if sample_token in nusc_annos:
                pass
                # nusc_annos[sample_token].extend(annos)
            else:
                nusc_annos[sample_token] = annos
        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path


    def evaluate_tracking(self,
                          results,
                          metric='bbox',
                          logger=None,
                          jsonfile_prefix=None,
                          result_names=['pts_bbox'],
                          show=False,
                          out_dir=None,
                          pipeline=None):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        result_files, tmp_dir, with_motion = self.format_tracking_results(results, jsonfile_prefix)

        if isinstance(result_files, dict):
            results_dict = dict()
            for name in result_names:                    
                print('Evaluating tracking bboxes of {}'.format(name))
                ret_dict = self._evaluate_tracking_single(result_files[name])
                results_dict.update(ret_dict)
                if with_motion:
                    print('Evaluating motion bboxes of {}'.format(name))
                    ret_dict = self._evaluate_motion_single(result_files[name])
                    results_dict.update(ret_dict)

        elif isinstance(result_files, str):
            results_dict = self._evaluate_tracking_single(result_files)
            if with_motion:
                print('Evaluating motion bboxes of')
                ret_dict = self._evaluate_motion_single(result_files)
                results_dict.update(ret_dict)

        if tmp_dir is not None:
            tmp_dir.cleanup()

        if show:
            self.show(results, out_dir, pipeline=pipeline)
        return results_dict

    def format_tracking_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a
                dict containing the json filepaths, `tmp_dir` is the temporal
                directory created for saving json files when
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) >= len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        # currently the output prediction results could be in two formats
        # 1. list of dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...)
        # 2. list of dict('pts_bbox' or 'img_bbox':
        #     dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...))
        # this is a workaround to enable evaluation of both formats on nuScenes
        # refer to https://github.com/open-mmlab/mmdetection3d/issues/449
        if not ('pts_bbox' in results[0] or 'img_bbox' in results[0]):
            result_files, with_motion = self._format_tracking_bbox(results, jsonfile_prefix)
        else:
            # should take the inner dict out of 'pts_bbox' or 'img_bbox' dict
            result_files = dict()
            for name in ['pts_bbox']:
                print(f'\nFormating tracking bboxes of {name}')
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_file, with_motion = self._format_tracking_bbox(results_, tmp_file_)
                result_files.update(
                    {name: result_file})
        return result_files, tmp_dir, with_motion

    def _format_tracking_bbox(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}
        mapped_class_names = self.CLASSES
        print('Start to convert tracking format...')
        processed_set = set()
        with_motion = False
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            boxes = det['boxes_3d'].tensor.numpy()
            # scores = det['scores_3d'].numpy()
            labels = det['labels_3d'].numpy()
            sample_id = det.get('index', sample_id)
            if 'track_scores' not in det:
                print('no tracking info')
                return None, with_motion
            tracking_scores = det['track_scores'].numpy()
            
            obj_idxes = det['obj_idxes'].numpy()
            if sample_id in processed_set: continue
            processed_set.add(sample_id)
            sample_token = self.data_infos[sample_id]['token']
  
            trans = self.data_infos[sample_id]['cams'][
                self.ego_cam]['ego2global_translation']
            rot = self.data_infos[sample_id]['cams'][
                self.ego_cam]['ego2global_rotation']
            rot = pyquaternion.Quaternion(rot)
            annos = list()

            for i, box in enumerate(boxes):
                if tracking_scores[i] < 0: continue
                name = mapped_class_names[labels[i]]
                if name not in self.TRACKING_CLASSES: continue
                center = box[:3]
                wlh = box[[4, 3, 5]]
                box_yaw = box[6]
                box_vel = box[7:].tolist()
                box_vel.append(0)
                quat =  pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw)
                nusc_box = NuScenesBox(center, wlh, quat, velocity=box_vel)
                nusc_box.rotate(rot)
                nusc_box.translate(trans)
                if np.sqrt(nusc_box.velocity[0]**2 +
                           nusc_box.velocity[1]**2) > 0.2:
                    if name in [
                            'car',
                            'construction_vehicle',
                            'bus',
                            'truck',
                            'trailer',
                    ]:
                        attr = 'vehicle.moving'
                    elif name in ['bicycle', 'motorcycle']:
                        attr = 'cycle.with_rider'
                    else:
                        attr = self.DefaultAttribute[name]
                else:
                    if name in ['pedestrian']:
                        attr = 'pedestrian.standing'
                    elif name in ['bus']:
                        attr = 'vehicle.stopped'
                    else:
                        attr = self.DefaultAttribute[name]
                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=nusc_box.center.tolist(),
                    size=nusc_box.wlh.tolist(),
                    rotation=nusc_box.orientation.elements.tolist(),
                    velocity=nusc_box.velocity[:2],
                    tracking_name=name,
                    detection_name=name,
                    detection_score=float(tracking_scores[i]),
                    attribute_name=attr,
                    tracking_score=float(tracking_scores[i]),
                    tracking_id=obj_idxes[i]
                )
                if 'motion_traj' in det:
                    with_motion = True
                    nusc_anno['traj'] = det['motion_traj'][i]
                    nusc_anno['traj_scores'] = det['motion_cls'][i]
                annos.append(nusc_anno)
            # other views results of the same frame should be concatenated
            if sample_token in nusc_annos:
                pass
                # nusc_annos[sample_token].extend(annos)
            else:
                nusc_annos[sample_token] = annos
        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc_tracking.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path, with_motion



    def _evaluate_motion_single(self,
                                  result_path,
                                  logger=None,
                                  metric='bbox',
                                  result_name='pts_bbox'):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        if result_path is None: return {}
        from nuscenes import NuScenes
        output_dir = osp.join(*osp.split(result_path)[:-1])
        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        from .evals.nuscenes_eval_motion import MotionEval
        if self.nusc is None:
            self.nusc = NuScenes(version=self.version, dataroot=self.data_root, verbose=False)
        self.nusc_eval_motion = MotionEval(
            self.nusc,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=True,
            data_infos=self.data_infos,
            ann_file=self.ann_file,
            category_convert_type='motion_category'
        )
        print('-'*50)
        print(
            'Evaluate on motion category, merge class for vehicles and pedestrians...')
        print('evaluate standard motion metrics...')
        self.nusc_eval_motion.main(
            plot_examples=0,
            render_curves=False,
            eval_mode='standard')
        print('evaluate motion mAP-minFDE metrics...')
        self.nusc_eval_motion.main(
            plot_examples=0,
            render_curves=False,
            eval_mode='motion_map')
        print('evaluate EPA motion metrics...')
        self.nusc_eval_motion.main(
            plot_examples=0,
            render_curves=False,
            eval_mode='epa')
        print('-'*50)
        print('Evaluate on detection category...')
        self.nusc_eval_motion = MotionEval(
            self.nusc,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=True,
            data_infos=self.data_infos,
            category_convert_type='detection_category'
        )
        print('evaluate standard motion metrics...')
        self.nusc_eval_motion.main(
            plot_examples=0,
            render_curves=False,
            eval_mode='standard')
        print('evaluate EPA motion metrics...')
        self.nusc_eval_motion.main(
            plot_examples=0,
            render_curves=False,
            eval_mode='motion_map')
        print('evaluate EPA motion metrics...')
        self.nusc_eval_motion.main(
            plot_examples=0,
            render_curves=False,
            eval_mode='epa')
        return {}
            
    def _evaluate_tracking_single(self,
                                  result_path,
                                  logger=None,
                                  metric='bbox',
                                  result_name='pts_bbox'):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        if result_path is None: return {}
        from nuscenes import NuScenes
        output_dir = osp.join(*osp.split(result_path)[:-1])

        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        from nuscenes.eval.tracking.evaluate import TrackingEval
        from nuscenes.eval.common.config import config_factory as track_configs

        cfg = track_configs("tracking_nips_2019")
        nusc_eval = TrackingEval(
            config=cfg,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=True,
            nusc_version=self.version,
            nusc_dataroot=self.data_root
        )
        metrics = nusc_eval.main()
        # record metrics
        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        print(metrics)
        detail = dict()
        metric_prefix = f'{result_name}_NuScenes'
        keys = ['amota', 'amotp', 'recall', 'motar',
                'gt', 'mota', 'motp', 'mt', 'ml', 'faf',
                'tp', 'fp', 'fn', 'ids', 'frag', 'tid', 'lgd']
        for key in keys:
            detail['{}/{}'.format(metric_prefix, key)] = metrics[key]
        return detail       

    def evaluate_occupancy(self, occ_results, runner=None, show_dir=None, save=False, **eval_kwargs):
        from .occ_metrics import Metric_mIoU, Metric_FScore
        if show_dir is not None:
            # import os
            # if not os.path.exists(show_dir):

            mmcv.mkdir_or_exist(show_dir)
            mmcv.mkdir_or_exist(os.path.join(show_dir, 'occupancy_pred'))
            print('\nSaving output and gt in {} for visualization.'.format(show_dir))
            begin= 0 # eval_kwargs.get('begin',None)

            end=1 if not save else len(occ_results) # eval_kwargs.get('end',None)
        self.occ_eval_metrics = Metric_mIoU(
            num_classes=18,
            use_lidar_mask=False,
            use_image_mask=True)
        
        self.eval_fscore = False
        if  self.eval_fscore:
            self.fscore_eval_metrics = Metric_FScore(
                leaf_size=10,
                threshold_acc=0.4,
                threshold_complete=0.4,
                voxel_size=[0.4, 0.4, 0.4],
                range=[-40, -40, -1, 40, 40, 5.4],
                void=[17, 255],
                use_lidar_mask=False,
                use_image_mask=True,
            )
        count = 0
        print('\nStarting Evaluation...')
        processed_set = set()
        for occ_pred_w_index in tqdm(occ_results):
            index = occ_pred_w_index['index']
            if index in processed_set: continue
            processed_set.add(index)

            occ_pred = occ_pred_w_index['pred_occupancy']
            info = self.data_infos[index]
            scene_name = info['scene_name']
            sample_token = info['token']
            occupancy_file_path = osp.join(self.occupancy_path, scene_name, sample_token, 'labels.npz')
            occ_gt = np.load(occupancy_file_path)
 
            gt_semantics = occ_gt['semantics']
            mask_lidar = occ_gt['mask_lidar'].astype(bool)
            mask_camera = occ_gt['mask_camera'].astype(bool)            
            # if show_dir is not None:
            #     if begin is not None and end is not None:
            #         if index>= begin and index<end:
            #             sample_token = info['token']
            #             count += 1
            #             save_path = os.path.join(show_dir, 'occupancy_pred', scene_name+'_'+sample_token)
            #             np.savez_compressed(save_path, pred=occ_pred[mask_camera], gt=occ_gt, sample_token=sample_token)
            #             with open(os.path.join(show_dir, 'occupancy_pred', 'file.txt'),'a') as f:
            #                 f.write(save_path+'\n')
                        # np.savez_compressed(save_path+'_gt', pred= occ_gt['semantics'], gt=occ_gt, sample_token=sample_token)
                # else:
                #     sample_token=info['token']
                #     save_path=os.path.join(show_dir,str(index).zfill(4))
                #     np.savez_compressed(save_path,pred=occ_pred,gt=occ_gt,sample_token=sample_token)


            self.occ_eval_metrics.add_batch(occ_pred[mask_camera], gt_semantics, mask_lidar, mask_camera)
            if self.eval_fscore:
                self.fscore_eval_metrics.add_batch(occ_pred[mask_camera], gt_semantics, mask_lidar, mask_camera)
   
        res = self.occ_eval_metrics.count_miou()
        if self.eval_fscore:
            res.update(self.fscore_eval_metrics.count_fscore())
        

        return res 
        
    def evaluate_mask(self, results):
        results_dict = {}
        iou = 0
        # ret_f1=[0,0,0,0,0]
        for i in range(len(results)):
            iou+=results[i]['iou']
        n=len(results)
        iou = iou/n
        results_dict['iou'] = iou
        return results_dict

    def evaluate_map(self,
                 results,
                 map_metric='chamfer',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pred_map'],
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """

        result_files, tmp_dir = self.format_map_results(results, jsonfile_prefix)

        if isinstance(result_files, dict):
            results_dict = dict()
            for name in result_names:
                print('Evaluating map of {}'.format(name))
                ret_dict = self._evaluate_map_single(result_files[name], map_metric=map_metric)
            results_dict.update(ret_dict)
        elif isinstance(result_files, str):
            results_dict = self._evaluate_map_single(result_files, map_metric=map_metric)

        if tmp_dir is not None:
            tmp_dir.cleanup()

        if show:
            self.show(results, out_dir, pipeline=pipeline)
        return results_dict


    def world2bev_vis(self, x, y):
            return int((x + 51.2) * 15), int((y + 51.2) * 15)

    def __map_visual__(self, gt_map, pred_map, index=0):
        
        import cv2
        for t, map_ in enumerate([gt_map, pred_map]):
            bev_img = np.ones([2000, 2000, 3], dtype=np.float32) * 255
            bev_img = bev_img.astype(np.float32)

            bev_img = cv2.circle(bev_img, self.world2bev_vis(0, 0), 5, (0, 255, 0), thickness=-1) 
            # from IPython import embed
            # embed()
            # exit()

            for k, line in enumerate(map_):
                label = line['type']
                score = line.get('confidence_level', 1)
                pts = line['pts']
                if score<0.3: continue
                corners = np.array([self.world2bev_vis(*pt) for pt in pts])
                corners = [each for each in corners if ((each>=0).all() & (each<2000).all())]
                colors = [(255, 255, 0), (255, 0, 0), (0, 255, 0)]
                for i, corner in enumerate(corners[:-1]):
                    bev_img = cv2.circle(bev_img, corners[i], 2, (61, 102, 255))
                    bev_img = cv2.line(bev_img, corners[i], corners[i+1], color=colors[label], thickness=1)
            mmcv.imwrite(bev_img, f'map_{index}_{t}.png')
        print('saved') 

    def _evaluate_map_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         map_metric='chamfer',
                         result_name='pts_bbox'):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        detail = dict()

        output_dir = osp.join(*osp.split(result_path)[:-1])

        from .map_utils.mean_ap import eval_map
        from .map_utils.mean_ap import format_res_gt_by_classes
        result_path = osp.abspath(result_path)
        
        print('Formating results & gts by classes')
        pred_results = mmcv.load(result_path)
        map_results = pred_results['map_results']
        gt_anns = mmcv.load(self.map_ann_file)
        map_annotations = gt_anns['GTs']
        cls_gens, cls_gts = format_res_gt_by_classes(result_path,
                                                     map_results,
                                                     map_annotations,
                                                     cls_names=self.MAPCLASSES,
                                                     num_pred_pts_per_instance=20,
                                                     eval_use_same_gt_sample_num_flag=True,
                                                     pc_range=self.pc_range)
        # for i in range(10):
        #     self.__map_visual__(map_annotations[i]['vectors'], map_results[map_annotations[i]['sample_token']]['vectors'], index=i)
        map_metrics = map_metric if isinstance(map_metric, list) else [map_metric]
        allowed_metrics = ['chamfer', 'iou']
        for metric in map_metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        for metric in map_metrics:
            print('-*'*10+f'use metric:{metric}'+'-*'*10)
            if metric == 'chamfer':
                thresholds = [0.5,1.0,1.5]
            elif metric == 'iou':
                thresholds= np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
            cls_aps = np.zeros((len(thresholds),self.NUM_MAPCLASSES))
            for i, thr in enumerate(thresholds):
                print('-*'*10+f'threshhold:{thr}'+'-*'*10)
                mAP, cls_ap = eval_map(
                                map_results,
                                map_annotations,
                                cls_gens,
                                cls_gts,
                                threshold=thr,
                                cls_names=self.MAPCLASSES,
                                logger=logger,
                                num_pred_pts_per_instance=20,
                                pc_range=self.pc_range,
                                metric=metric)
                for j in range(self.NUM_MAPCLASSES):
                    cls_aps[i, j] = cls_ap[j]['ap']
            for i, name in enumerate(self.MAPCLASSES):
                print('{}: {}'.format(name, cls_aps.mean(0)[i]))
                detail['NuscMap_{}/{}_AP'.format(metric,name)] =  cls_aps.mean(0)[i]
            print('map: {}'.format(cls_aps.mean(0).mean()))
            detail['NuscMap_{}/mAP'.format(metric)] = cls_aps.mean(0).mean()
            for i, name in enumerate(self.MAPCLASSES):
                for j, thr in enumerate(thresholds):
                    if metric == 'chamfer':
                        detail['NuscMap_{}/{}_AP_thr_{}'.format(metric,name,thr)]=cls_aps[j][i]
                    elif metric == 'iou':
                        if thr == 0.5 or thr == 0.75:
                            detail['NuscMap_{}/{}_AP_thr_{}'.format(metric,name,thr)]=cls_aps[j][i]

        return detail
    

    def evaluate_bbox(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix='test',
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str], optional): Metrics to be evaluated.
                Default: 'bbox'.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str, optional): The prefix of json files including
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool, optional): Whether to visualize.
                Default: False.
            out_dir (str, optional): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)


        if isinstance(result_files, dict):
            results_dict = dict()
            for name in result_names:
                print('Evaluating bboxes of {}'.format(name))
                ret_dict = self._evaluate_single(result_files[name])
            results_dict.update(ret_dict)
        elif isinstance(result_files, str):
            results_dict = self._evaluate_single(result_files)

        if tmp_dir is not None:
            tmp_dir.cleanup()

        if show or out_dir:
            self.show(results, out_dir, show=show, pipeline=pipeline)

        return results_dict

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadPointsFromMultiSweeps',
                sweeps_num=10,
                file_client_args=dict(backend='disk')),
            dict(
                type='DefaultFormatBundle3D',
                class_names=self.CLASSES,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ]
        return Compose(pipeline)

    def show(self, results, out_dir, show=False, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Whether to visualize the results online.
                Default: False.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        for i, result in enumerate(results):
            if 'pts_bbox' in result.keys():
                result = result['pts_bbox']
            data_info = self.data_infos[i]
            pts_path = data_info['lidar_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            points = self._extract_data(i, pipeline, 'points').numpy()
            # for now we convert points into depth mode
            points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
                                               Coord3DMode.DEPTH)
            inds = result['scores_3d'] > 0.1
            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d'].tensor.numpy()
            show_gt_bboxes = Box3DMode.convert(gt_bboxes, Box3DMode.LIDAR,
                                               Box3DMode.DEPTH)
            pred_bboxes = result['boxes_3d'][inds].tensor.numpy()
            show_pred_bboxes = Box3DMode.convert(pred_bboxes, Box3DMode.LIDAR,
                                                 Box3DMode.DEPTH)
            show_result(points, show_gt_bboxes, show_pred_bboxes, out_dir,
                        file_name, show)


def output_to_nusc_box(detection, with_velocity=True):
    """Convert the output to the box class in the nuScenes.

    Args:
        detection (dict): Detection results.

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.

    Returns:
        list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
    """
    box3d = detection['boxes_3d']
    scores = detection['scores_3d'].numpy()
    labels = detection['labels_3d'].numpy()

    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()

    # our LiDAR coordinate system -> nuScenes box coordinate system
    nus_box_dims = box_dims[:, [1, 0, 2]]

    box_list = []
    for i in range(len(box3d)):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        if with_velocity:
            velocity = (*box3d.tensor[i, 7:9], 0.0)
        else:
            velocity = (0, 0, 0)
        # velo_val = np.linalg.norm(box3d[i, 7:9])
        # velo_ori = box3d[i, 6]
        # velocity = (
        # velo_val * np.cos(velo_ori), velo_val * np.sin(velo_ori), 0.0)
        box = NuScenesBox(
            box_gravity_center[i],
            nus_box_dims[i],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity)
        box_list.append(box)
    return box_list

@DATASETS.register_module()
class NuscenesOccupancy(NuScenesDataset):

    CLASSES = [
        "empty",
        "barrier",
        "bicycle",
        "bus",
        "car",
        "construction",
        "motorcycle",
        "pedestrian",
        "trafficcone",
        "trailer",
        "truck",
        "driveable_surface",
        "other",
        "sidewalk",
        "terrain",
        "mannade",
        "vegetation",
    ]

    def __init__(self, occupancy_info='data/nuscenes/occupancy_category.json', **kwargs):

        super().__init__(**kwargs)
        self.CLASSES = [
            "empty",
            "barrier",
            "bicycle",
            "bus",
            "car",
            "construction",
            "motorcycle",
            "pedestrian",
            "trafficcone",
            "trailer",
            "truck",
            "driveable_surface",
            "other",
            "sidewalk",
            "terrain",
            "mannade",
            "vegetation",
        ]

        self.occupancy_info = mmcv.load(occupancy_info)

    def get_cat_ids(self, idx):
        """Get category distribution of single scene.

        Args:
            idx (int): Index of the data_info.

        Returns:
            dict[list]: for each category, if the current scene
                contains such boxes, store a list containing idx,
                otherwise, store empty list.
        """
        info = self.data_infos[idx]

        token = info['token']
        category = self.occupancy_info[token]
        cat_ids = []
        for k, v in category.items():
            k = int(k)
            if k == 17: continue
            logv = max((np.log(v)/np.log(100)).round(),1)
            cat_ids.extend([k] * int(logv))
        return cat_ids

def lidar_nusc_box_to_global(info,
                             boxes,
                             classes,
                             eval_configs,
                             eval_version='detection_cvpr_2019'):
    """Convert the box from ego to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs (object): Evaluation configuration object.
        eval_version (str, optional): Evaluation version.
            Default: 'detection_cvpr_2019'

    Returns:
        list: List of standard NuScenesBoxes in the global
            coordinate.
    """
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(pyquaternion.Quaternion(info['lidar2ego_rotation']))
        box.translate(np.array(info['lidar2ego_translation']))
        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to global coord system
        box.rotate(pyquaternion.Quaternion(info['ego2global_rotation']))
        box.translate(np.array(info['ego2global_translation']))
        box_list.append(box)
    return box_list



def invert_matrix_egopose_numpy(egopose):
    """ Compute the inverse transformation of a 4x4 egopose numpy matrix."""
    inverse_matrix = np.zeros((4, 4), dtype=np.float32)
    rotation = egopose[:3, :3]
    translation = egopose[:3, 3]
    inverse_matrix[:3, :3] = rotation.T
    inverse_matrix[:3, 3] = -np.dot(rotation.T, translation)
    inverse_matrix[3, 3] = 1.0
    return inverse_matrix

def convert_egopose_to_matrix_numpy(rotation, translation):
    transformation_matrix = np.zeros((4, 4), dtype=np.float32)
    transformation_matrix[:3, :3] = rotation
    transformation_matrix[:3, 3] = translation
    transformation_matrix[3, 3] = 1.0
    return transformation_matrix


def output_to_vecs(detection):
    # box3d = detection['map_boxes_3d'].numpy()
    scores = detection['map_scores_3d'].numpy()
    labels = detection['map_labels_3d'].numpy()
    pts = detection['map_pts_3d'].numpy()

    vec_list = []
    # import pdb;pdb.set_trace()
    for i in range(pts.shape[0]):
        vec = dict(
            bbox =[], # box3d[i], # xyxy
            label=labels[i],
            score=scores[i],
            pts=pts[i],
        )
        vec_list.append(vec)
    return vec_list