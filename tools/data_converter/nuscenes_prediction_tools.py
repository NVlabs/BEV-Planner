# ------------------------------------------------------------------------
# Copyright (c) 2023 toyota research instutute.
# ------------------------------------------------------------------------
# Modified from FutureDet (https://github.com/neeharperi/FutureDet)
# ------------------------------------------------------------------------
import numpy as np
from pyquaternion import Quaternion
from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from itertools import tee
from copy import deepcopy


def get_forecasting_annotations(nusc: NuScenes, annotations, length):
    """Acquire the trajectories for each box
    """
    forecast_annotations = []
    forecast_boxes = []   
    forecast_trajectory_type = []
    forecast_visibility_mask = []
    sample_tokens = [s["token"] for s in nusc.sample]

    for annotation in annotations:
        tracklet_box = []
        tracklet_annotation = []
        tracklet_visiblity_mask = []
        tracklet_trajectory_type = []

        token = nusc.sample[sample_tokens.index(annotation["sample_token"])]["data"]["LIDAR_TOP"]
        sd_record = nusc.get("sample_data", token)
        cs_record = nusc.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
        pose_record = nusc.get("ego_pose", sd_record["ego_pose_token"])

        visibility = True
        for step in range(length):
            box = Box(center = annotation["translation"],
                      size = annotation["size"],
                      orientation = Quaternion(annotation["rotation"]),
                      velocity = nusc.box_velocity(annotation["token"]),
                      name = annotation["category_name"],
                      token = annotation["token"])
            
            # move box to the ego-system when the prediction is made
            box.translate(-np.array(pose_record["translation"]))
            box.rotate(Quaternion(pose_record["rotation"]).inverse)

            #  Move box to sensor coord system
            box.translate(-np.array(cs_record["translation"]))
            box.rotate(Quaternion(cs_record["rotation"]).inverse)

            tracklet_box.append(box)
            tracklet_annotation.append(annotation)
            tracklet_visiblity_mask.append(visibility)

            next_token = annotation['next']
            if next_token != '':
                annotation = nusc.get('sample_annotation', next_token)
            else:
                # if the trajectory cannot be prolonged anymore,
                # use the last one to pad and set the visibility flag
                annotation = annotation
                visibility = False
    
        tokens = [b["sample_token"] for b in tracklet_annotation]
        time = [get_time(nusc, src, dst) for src, dst in window(tokens, 2)]
        tracklet_trajectory_type = trajectory_type(nusc, tracklet_box, time, length) # same as FutureDet

        forecast_boxes.append(tracklet_box)
        forecast_annotations.append(tracklet_annotation)
        forecast_trajectory_type.append(length * [tracklet_trajectory_type])
        forecast_visibility_mask.append(tracklet_visiblity_mask)
    return forecast_boxes, forecast_annotations, forecast_visibility_mask, forecast_trajectory_type


def window(iterable, size):
    iters = tee(iterable, size)
    for i in range(1, size):
        for each in iters[i:]:
            next(each, None)

    return zip(*iters)

def get_time(nusc, src_token, dst_token):
    time_last = 1e-6 * nusc.get('sample', src_token)["timestamp"]
    time_first = 1e-6 * nusc.get('sample', dst_token)["timestamp"]
    time_diff = time_first - time_last

    return time_diff 


def center_distance(gt_box, pred_box) -> float:
    """
    L2 distance between the box centers (xy only).
    :param gt_box: GT annotation sample.
    :param pred_box: Predicted sample.
    :return: L2 distance.
    """
    return np.linalg.norm(np.array(pred_box.center[:2]) - np.array(gt_box.center[:2]))


def trajectory_type(nusc, boxes, time, timesteps=7, past=False):
    target = boxes[-1]
    
    static_forecast = deepcopy(boxes[0])

    linear_forecast = deepcopy(boxes[0])
    vel = linear_forecast.velocity[:2]
    disp = np.sum(list(map(lambda x: np.array(list(vel) + [0]) * x, time)), axis=0)

    if past:
        linear_forecast.center = linear_forecast.center - disp

    else:
        linear_forecast.center = linear_forecast.center + disp
    
    if center_distance(target, static_forecast) < max(target.wlh[0], target.wlh[1]):
        # return "static"
        return 0

    elif center_distance(target, linear_forecast) < max(target.wlh[0], target.wlh[1]):
        # return "linear"
        return 1

    else:
        # return "nonlinear"
        return 2