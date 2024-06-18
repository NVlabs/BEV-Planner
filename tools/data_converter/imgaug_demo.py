

#!usr/bin/python
# -*- coding: utf-8 -*-


import cv2
import random
import os
import os.path as osp
from matplotlib import pyplot as plt
# import albumentations as A
from imgaug import augmenters as iaa
from nuscenes import NuScenes
from nuscenes.utils import splits
fog_aug = iaa.Fog()
snow_aug = iaa.Snowflakes(flake_size=(0.7, 0.95), speed=(0.001, 0.03))
rain_aug = iaa.Rain(drop_size=(0.10, 0.20))
noise_aug = iaa.imgcorruptlike.GaussianNoise(severity=1)
# transform = A.Compose(
#     [A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0.5, p=1)],
# )
import mmcv 
def get_available_scenes(nusc):
    """Get available scenes from the input nuscenes class.

    Given the raw data, get the information of available scenes for
    further info generation.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.

    Returns:
        available_scenes (list[dict]): List of basic information for the
            available scenes.
    """
    available_scenes = []
    print('total scene num: {}'.format(len(nusc.scene)))
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path:
                # path from lyftdataset is absolute path
                lidar_path = lidar_path.split(f'{os.getcwd()}/')[-1]
                # relative path
            if not mmcv.is_filepath(lidar_path):
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print('exist scene num: {}'.format(len(available_scenes)))
    return available_scenes

VERSION= 'v1.0-trainval'
NUSCENES = 'nuscenes'
nuscenes_version = VERSION
dataroot = f'./data/{NUSCENES}/'
nuscenes = NuScenes(nuscenes_version, dataroot)
val_scenes = splits.val
# filter existing scenes.
available_scenes = get_available_scenes(nuscenes)
available_scene_names = [s['name'] for s in available_scenes]
val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
val_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in val_scenes
   ])

val_imgs = set()
for sample in mmcv.track_iter_progress(nuscenes.sample):
   camera_types = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
   ]
   if sample['scene_token'] in val_scenes:
      for cam in camera_types:
         cam_token = sample['data'][cam]
         cam_path, _, cam_intrinsic = nuscenes.get_sample_data(cam_token)
         val_imgs.add(cam_path.split('/')[-1])


aug_mapper = dict(
   fog=iaa.Fog(),
   snow=iaa.Snowflakes(flake_size=(0.7, 0.95), speed=(0.001, 0.03)),
   rain=iaa.Rain(drop_size=(0.10, 0.20)),
   noise=iaa.imgcorruptlike.GaussianNoise(severity=1)
)
#imgaug test

ori_sample_path = '/mount/data/FBBEV/data/nuscenes/samples'
det_sample_path = '/mount/data/FBBEV/data/nuscenes_aug/samples_rain'
cams = os.listdir(det_sample_path)
for cam in cams:
   imgs = os.listdir(osp.join(ori_sample_path, cam))
   for img_name in imgs:
      imglist=[]
      if img_name not in val_imgs: continue
      img_path = osp.join(ori_sample_path, cam, img_name)
      print(img_path)
      img = cv2.imread(img_path)
      img = cv2.resize(img, (800, 450))
      imglist.append(img)
      augs = ['noise']# ['fog', 'rain', 'snow', 'noise']
      for aug_key in augs:
         
         seq = iaa.Sequential([
             aug_mapper[aug_key]
         ])
         images_aug = seq.augment_images(imglist)
         images_aug = cv2.resize(images_aug[0], (1600, 900))
         # print(f'/mount/data/FBBEV/data/nuscenes_aug/samples_{aug_key}/{cam}/{img_name}')
         cv2.imwrite(f'/mount/data/FBBEV/data/nuscenes_aug/samples_{aug_key}/{cam}/{img_name}', images_aug)


    #   images_aug = transform(image=img)['image']
    #   images_aug = cv2.resize(images_aug, (1600, 900))
    #   cv2.imwrite(f'/mount/data/FBBEV/data/nuscenes_aug/samples_sun/{cam}/{img_name}', images_aug)




