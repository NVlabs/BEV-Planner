# ------------------------------------------------------------------------
# Copyright (c) 2023 toyota research instutute.
# ------------------------------------------------------------------------
from .instances import Instances
import torch
import numpy as np


class RunTimeTracker:
    def __init__(self, output_threshold=0.2, score_threshold=0.4, 
                       max_age_since_update=1, **kwargs):
        self.current_id = 1
        self.current_seq = 0
        self.timestamp = None
        self.time_delta = None
        self.query_embeddings = None
        self.reference_points = None
        self.frame_index = 0
    
        self.track_instances = None
        self.timestamp = None
        self.first_frame = None

        self.threshold = score_threshold
        self.output_threshold = output_threshold
        self.max_age_since_update = max_age_since_update
    
    def update_active_tracks(self, track_instances, active_mask):
        # first frame
        if self.track_instances is None:
            self.track_instances = track_instances[active_mask]
            return
        
        live_mask = torch.zeros_like(track_instances.obj_idxes).bool().detach()
        for i in range(len(track_instances)):
            if active_mask[i]:
                track_instances.disappear_time[i] = 0
                live_mask[i] = True
            elif track_instances.track_query_mask[i]:
                track_instances.disappear_time[i] += 1
                if track_instances.disappear_time[i] < self.max_age_since_update:
                    live_mask[i] = True
        self.track_instances = track_instances[live_mask]
        return
    
    def get_active_mask(self, track_instances, training=True):
        if training:
            active_mask = (track_instances.matched_gt_idxes >= 0)
        return active_mask
    
    def empty(self):
        """Copy the historical buffer parts from the init
        """
        self.current_id = 1
        self.current_seq = 0
        self.timestamp = None
        self.query_embeddings = None
        self.reference_points = None
        self.frame_index = 0
    
        self.track_instances = None
        self.timestamp = None
        self.first_frame = None