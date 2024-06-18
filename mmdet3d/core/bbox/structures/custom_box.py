# nuScenes dev-kit.
# Code written by Oscar Beijbom, 2018.

import copy
import os.path as osp
import struct
from abc import ABC, abstractmethod
from functools import reduce
from typing import Tuple, List, Dict

import cv2
import numpy as np
from matplotlib.axes import Axes
from pyquaternion import Quaternion

from nuscenes.lidarseg.lidarseg_utils import colormap_to_colors, create_lidarseg_legend
from nuscenes.utils.data_io import load_bin_file
from nuscenes.utils.geometry_utils import view_points, transform_matrix
from nuscenes.utils.data_classes import Box



class CustomBox(Box):
    """ Simple data class representing a 3d box including, label, score and velocity. """

    def __init__(self,
                 center: List[float],
                 size: List[float],
                 orientation: Quaternion,
                 label: int = np.nan,
                 score: float = np.nan,
                 velocity: Tuple = (np.nan, np.nan, np.nan),
                 name: str = None,
                 token: str = None):
        """
        :param center: Center of box given as x, y, z.
        :param size: Size of box in width, length, height.
        :param orientation: Box orientation.
        :param label: Integer label, optional.
        :param score: Classification score, optional.
        :param velocity: Box velocity in x, y, z direction.
        :param name: Box name, optional. Can be used e.g. for denote category name.
        :param token: Unique string identifier from DB.
        """
        # assert not np.any(np.isnan(center))
        # assert not np.any(np.isnan(size))
        # assert len(center) == 3
        # assert len(size) == 3
        # assert type(orientation) == Quaternion

        # self.center = np.array(center)
        # self.wlh = np.array(size)
        # self.orientation = orientation
        # self.label = int(label) if not np.isnan(label) else label
        # self.score = float(score) if not np.isnan(score) else score
        # self.velocity = np.array(velocity)
        # self.name = name
        # self.token = token

        super().__init__(
            center=center,
            size=size,
            orientation=orientation,
            label=label,
            score = score,
            velocity = velocity,
            name = name,
            token = token
        )

    def render(self,
               axis: Axes,
               view: np.ndarray = np.eye(3),
               normalize: bool = False,
               colors: Tuple = ('b', 'r', 'k'),
               linewidth: float = 2) -> None:
        """
        Renders the box in the provided Matplotlib axis.
        :param axis: Axis onto which the box should be drawn.
        :param view: <np.array: 3, 3>. Define a projection in needed (e.g. for drawing projection in an image).
        :param normalize: Whether to normalize the remaining coordinate.
        :param colors: (<Matplotlib.colors>: 3). Valid Matplotlib colors (<str> or normalized RGB tuple) for front,
            back and sides.
        :param linewidth: Width in pixel of the box sides.
        """
        corners = view_points(self.corners(), view, normalize=normalize)[:2, :]

        def draw_rect(selected_corners, color):
            prev = selected_corners[-1]
            for corner in selected_corners:
                axis.plot([prev[0], corner[0]], [prev[1], corner[1]], color=color, linewidth=linewidth)
                prev = corner

        # Draw the sides
        for i in range(4):
            axis.plot([corners.T[i][0], corners.T[i + 4][0]],
                      [corners.T[i][1], corners.T[i + 4][1]],
                      color=colors[2], linewidth=linewidth)

        # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
        draw_rect(corners.T[:4], colors[0])
        draw_rect(corners.T[4:], colors[1])

        # Draw line indicating the front
        center_bottom_forward = np.mean(corners.T[2:4], axis=0)
        center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
        axis.plot([center_bottom[0], center_bottom_forward[0]],
                  [center_bottom[1], center_bottom_forward[1]],
                  color=colors[0], linewidth=linewidth)
        # from IPython import embed
        # embed()
        # exit()  
        #   
        # In [1]: corners.T
        # Out[1]: 
        # array([[135.10084664, 217.64073984],
        #     [145.04250652, 217.12554824],
        #     [145.04250652, 217.12554824],
        #     [135.10084664, 217.64073984],
        #     [134.95749348, 214.87445176],
        #     [144.89915336, 214.35926016],
        #     [144.89915336, 214.35926016],
        #     [134.95749348, 214.87445176]])  
        x_coords, y_coords = zip(*corners.T[[0,1,6,7]])
        # axis.fill(x_coords, y_coords, colors[0], alpha=0.8)

