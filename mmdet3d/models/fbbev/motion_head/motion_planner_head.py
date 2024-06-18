import torch
import torch.nn as nn 
from mmcv.cnn import Linear, bias_init_with_prob, Scale

from mmcv.runner import force_fp32
from mmdet.core import (build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet.models.utils import build_transformer
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet3d.core.bbox.coders import build_bbox_coder
# from .streampetr_utils import *
import copy
from mmdet.models.utils import NormedLinear
from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from mmdet3d.models.fbbev.utils import save_tensor
from mmcv.runner.base_module import BaseModule
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from ..streampetr.streampetr_utils import *
from ..planner_head.metric_stp3 import PlanningMetric

def get_ego_pos(points, pc_range):
    if points.size(-1) == 3:
        points = points * (pc_range[3:6] - pc_range[0:3]) + pc_range[0:3]
    elif  points.size(-1) == 2:
        points = points * (pc_range[3:5] - pc_range[0:2]) + pc_range[0:2]
    return points

def get_rel_pos(points, pc_range):
    if points.size(-1) == 3:
        return (points - pc_range[0:3]) / (pc_range[3:6] - pc_range[0:3])
    elif  points.size(-1) == 2:
        return (points - pc_range[0:2]) / (pc_range[3:5] - pc_range[0:2])


@HEADS.register_module()
class MotionPlannerHead(BaseModule):
    """Implements the DETR transformer head.
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        num_classes (int): Number of categories excluding the background.
        in_channels (int): Number of channels in the input feature map.
        num_query (int): Number of query in Transformer.
        num_reg_fcs (int, optional): Number of fully-connected layers used in
            `FFN`, which is then used for the regression head. Default 2.
        transformer (obj:`mmcv.ConfigDict`|dict): Config for transformer.
            Default: None.
        sync_cls_avg_factor (bool): Whether to sync the avg_factor of
            all ranks. Default to False.
        positional_encoding (obj:`mmcv.ConfigDict`|dict):
            Config for position encoding.
        loss_cls (obj:`mmcv.ConfigDict`|dict): Config of the
            classification loss. Default `CrossEntropyLoss`.
        loss_bbox (obj:`mmcv.ConfigDict`|dict): Config of the
            regression loss. Default `L1Loss`.
        loss_iou (obj:`mmcv.ConfigDict`|dict): Config of the
            regression iou loss. Default `GIoULoss`.
        tran_cfg (obj:`mmcv.ConfigDict`|dict): Training config of
            transformer head.
        test_cfg (obj:`mmcv.ConfigDict`|dict): Testing config of
            transformer head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """
    _version = 2

    def __init__(self,
                 # num_classes=1,
                 in_channels=256,
                 stride=[16],
                 embed_dims=256,
                 num_query=6,
                 num_reg_fcs=2,
                 memory_len=12,
                 topk_proposals=4,
                 num_propagated=0,
                 with_dn=True,
                 with_ego_pos=True,
                 match_with_velo=True,
                 match_costs=None,
                 transformer=None,
                 sync_cls_avg_factor=False,
                 code_weights=None,
                 bbox_coder=None,
                 loss_traj=dict(type='L1Loss', loss_weight=0.25),
                 init_cfg=None,
                 normedlinear=False,
                 point_cloud_range=None,
                 agent_decoder=dict(),
                 agent_map_decoder=dict(),
                 map_layer_index = -1,

                # planner
                 loss_plan_reg=dict(type='L1Loss', loss_weight=5.0),
                 loss_plan_bound=dict(type='PlanMapBoundLoss', loss_weight=1.0, dis_thresh=5.0),
                 loss_plan_col=dict(type='PlanCollisionLoss', loss_weight=5.0),
                 loss_plan_dir=dict(type='PlanMapDirectionLoss', loss_weight=2.5),
                 ego_agent_decoder = dict(
                    type='CustomTransformerDecoder',
                    num_layers=1,
                    return_intermediate=False,
                    transformerlayers=dict(
                        type='BaseTransformerLayer',
                        batch_first=True,
                        attn_cfgs=dict(
                            type='MotionSelfAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1,
                            dist_func_type='MDE',
                            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                            consider_map_quality=False,
                        ),
                        feedforward_channels=2048,
                        ffn_dropout=0.1,
                        operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
                 ego_map_decoder = dict(
                    type='CustomTransformerDecoder',
                    num_layers=1,
                    return_intermediate=False,
                    transformerlayers=dict(
                        type='BaseTransformerLayer',
                        batch_first=True,
                        attn_cfgs=dict(
                            type='MotionSelfAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1,
                            dist_func_type='MDE',
                            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                            consider_map_quality=True,
                        ),
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
                 ego_ego_decoder = dict(
                    type='CustomTransformerDecoder',
                    num_layers=1,
                    return_intermediate=False,
                    transformerlayers=dict(
                        type='BaseTransformerLayer',
                        batch_first=True,
                        attn_cfgs=dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            attn_drop=0.1,
                            proj_drop=0.1,
                        ),
                        feedforward_channels=1024,
                        ffn_dropout=0.1,
                        operation_order=('cross_attn', 'norm', 'ffn', 'norm'))),
                 **kwargs):

        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 2
        # if code_weights is not None:
        #     self.code_weights = code_weights
        # else:
        #     self.code_weights = [1.0, 1.0] # x, y, v_x, v_y
        # self.code_weights = self.code_weights[:self.code_size]


        self.traj_num_cls = 1

        self.num_query = num_query
        self.in_channels = in_channels
        self.num_reg_fcs = num_reg_fcs
        # self.train_cfg = train_cfg
        # self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.embed_dims = embed_dims
        self.map_layer_index = map_layer_index


        super(MotionPlannerHead, self).__init__()

        self.loss_traj = build_loss(loss_traj)
        self.log_softmax = nn.LogSoftmax(dim=2)
        # self.code_weights = nn.Parameter(torch.tensor(
        #     self.code_weights), requires_grad=False)
        self.pc_range = nn.Parameter(torch.tensor(
            point_cloud_range), requires_grad=False)

        self.fut_steps = 8
        self.num_fut_mode = 6


        self.agent_decoder = build_transformer_layer_sequence(agent_decoder)
        self.agent_map_decoder = build_transformer_layer_sequence(agent_map_decoder)

        self._init_layers()
        self.count = 0

        # planner
        self.ego_ego_decoder = build_transformer_layer_sequence(ego_ego_decoder)
        self.ego_agent_decoder = build_transformer_layer_sequence(ego_agent_decoder)
        self.ego_map_decoder = build_transformer_layer_sequence(ego_map_decoder)
        self.ego_fut_steps = 6
        self.ego_fut_mode = 3
        self.memory_len = 4
        self.loss_plan_reg = build_loss(loss_plan_reg)
        loss_plan_bound.update(point_cloud_range=point_cloud_range)
        loss_plan_col.update(point_cloud_range=point_cloud_range)
        loss_plan_dir.update(point_cloud_range=point_cloud_range)
        self.loss_plan_bound = build_loss(loss_plan_bound)
        self.loss_plan_col = build_loss(loss_plan_col)
        self.loss_plan_dir = build_loss(loss_plan_dir)
        self.ego_info = MLN(3)
        self._init_planer_layers()
        self.memory_traj = None
        self.planning_metric = PlanningMetric()
        self.count = 0

    def _init_planer_layers(self):
        """Initialize layers of the transformer head."""

        ego_fut_decoder = []
        ego_fut_dec_in_dim = self.embed_dims*2
        for _ in range(self.num_reg_fcs):
            ego_fut_decoder.append(Linear(ego_fut_dec_in_dim, ego_fut_dec_in_dim))
            ego_fut_decoder.append(nn.ReLU())
        ego_fut_decoder.append(Linear(ego_fut_dec_in_dim, self.ego_fut_mode*self.ego_fut_steps*2))
        self.ego_fut_decoder = nn.Sequential(*ego_fut_decoder)


        self.query_embedding = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

        self.motion_query_mlp = nn.Sequential(
            nn.Linear(2 * self.embed_dims * self.num_fut_mode , self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

        self.query_feat_embedding = nn.Embedding(1, self.embed_dims)
        self.memory_ego_embed = None
        self.time_embedding = nn.Embedding(self.memory_len, self.embed_dims)
        self.hist_ego_mlp = nn.Sequential(
            nn.Linear(self.embed_dims * 2, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

    def _init_layers(self):
        """Initialize layers of the transformer head."""

        traj_branch = []
        for _ in range(self.num_reg_fcs):
            traj_branch.append(Linear(self.embed_dims*2, self.embed_dims*2))
            traj_branch.append(nn.ReLU())
        traj_branch.append(Linear(self.embed_dims*2, self.fut_steps*self.code_size))
        traj_branch = nn.Sequential(*traj_branch)

        traj_cls_branch = []
        for _ in range(self.num_reg_fcs):
            traj_cls_branch.append(Linear(self.embed_dims*2, self.embed_dims*2))
            traj_cls_branch.append(nn.LayerNorm(self.embed_dims*2))
            traj_cls_branch.append(nn.ReLU(inplace=True))
        traj_cls_branch.append(Linear(self.embed_dims*2, self.traj_num_cls))
        traj_cls_branch = nn.Sequential(*traj_cls_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        motion_num_pred = 2
        self.traj_branches = _get_clones(traj_branch, motion_num_pred)
        self.traj_cls_branches = _get_clones(traj_cls_branch, motion_num_pred)

        # self.reference_points = nn.Embedding(self.num_query, 3)
        self.agent_info = MLN(17)
        self.agent_info_embedding = nn.Sequential(
            nn.Linear(17, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

        self.traj_mode_embedding = nn.Embedding(self.num_fut_mode, self.embed_dims)

    def pre_update_memory(self, data, fut_traj_from_velo):

        x = 1-data['start_of_sequence'] # original prev_exist, so we need do `not`
        B = x.size(0)
        # refresh the memory when the scene changes
        if self.memory_traj is None:
            self.memory_traj =  fut_traj_from_velo.unsqueeze(1).repeat(1, self.memory_len, 1, 1) # * 0
            self.memory_ego_embed = x.new_zeros(B, self.memory_len, self.embed_dims * 2)
        else:
            self.memory_traj = transform_reference_points(self.memory_traj, data['ego_pose_inv'], reverse=False)[..., :2]
            self.memory_traj = memory_refresh(self.memory_traj[:, :self.memory_len], x) 
            for i in range(B):
                # do not leak velo info，init all zeros
                if not x[i]: self.memory_traj[i, 0] = fut_traj_from_velo[i] * 0
            
            self.memory_ego_embed = memory_refresh(self.memory_ego_embed[:, :self.memory_len], x)

    def post_update_memory(self, data, ego_fut_trajs, ego_embeds):
        self.memory_traj = torch.cat([ego_fut_trajs, self.memory_traj], dim=1)
        self.memory_traj = torch.cat([self.memory_traj, torch.zeros_like(self.memory_traj[..., :1])], -1)
        self.memory_traj = transform_reference_points(self.memory_traj, data['ego_pose'], reverse=False)
        self.memory_ego_embed = torch.cat([ego_embeds, self.memory_ego_embed], dim=1).detach()
    

    def forward(self, 
            agent_instances,
            preds_map_dicts,
            img_metas=None,
            gt_ego_lcf_feat=None,
            gt_ego_fut_cmd=None,
            gt_ego_his_traj=None,
            gt_ego_fut_trajs=None):
        
        valid_length = [(matched_gt_idxes>=0).sum() for matched_gt_idxes in agent_instances.matched_gt_idxes]

        max_valid_query = max(valid_length)
        
        assert 0<=max_valid_query<=250
        agent_instances = agent_instances[:, :max_valid_query]
        agent_queries = agent_instances.query_feats
        agent_reference_points = agent_instances.reference_points
        mode_embedding = self.traj_mode_embedding.weight
        hist_mask = agent_instances.hist_mask
        B = len(agent_instances)
        hist_xyz_delta = (agent_instances.hist_xyz[:, :, 1:] - agent_instances.hist_xyz[:, :, :-1]) * hist_mask[:,:, :-1, None]
        agent_hist_info = torch.cat([hist_xyz_delta.flatten(-2, -1), agent_instances.hist_velo.flatten(-2, -1)], -1).detach()
        
        # I do believe this agent history infomation can be helpfull, so I use it twice
        agent_queries = self.agent_info(agent_queries, agent_hist_info)
        extra_agent_infos = (self.agent_info_embedding(agent_hist_info)[:, :, None, :].repeat(1, 1, self.num_fut_mode, 1)).flatten(1, 2)

        agent_queries = (agent_queries[:, :, None, :] + mode_embedding[None, None, :, :]).flatten(1, 2)

        hist_traj_points = agent_instances.hist_xyz.unsqueeze(2).repeat(1, 1, self.num_fut_mode, 1, 1).flatten(1, 2)
        hist_agent_xy = agent_instances.reference_points[:, :, :2].unsqueeze(2).repeat(1, 1, self.num_fut_mode, 1).flatten(1, 2)

        agent_queries = self.agent_decoder(agent_queries, reference_points_q=hist_traj_points, reference_points_v=hist_traj_points, pc_range=self.pc_range)
        
        map_queries = preds_map_dicts['queries'].clone()
        map_lines = preds_map_dicts['lines'].clone()
        map_scores = preds_map_dicts['scores'].clone()
        B, NMQ, K2 = map_lines.shape
        map_lines = map_lines.reshape(B, NMQ, K2//2, 2)
        map_pos = self.query_embedding(bevpos2posemb(map_lines.mean(-2)))
        map_lines = get_ego_pos(map_lines, self.pc_range)

        co_agent_queries = torch.cat([agent_queries, extra_agent_infos], -1)
        pred_traj_cls = self.traj_cls_branches[0](co_agent_queries).view(B, max_valid_query, self.num_fut_mode)
        pred_traj_cls = self.log_softmax(pred_traj_cls)
        pred_traj = self.traj_branches[0](co_agent_queries)
        B, N, PK = pred_traj.shape
        pred_traj = pred_traj.view(B, N, PK//self.code_size, self.code_size)

        fut_traj_points = torch.cat([hist_agent_xy.unsqueeze(-2), pred_traj[..., :2]], -2)
        fut_traj_points = torch.cumsum(fut_traj_points, -2)[:, :, 1:]

        agent_queries = self.agent_map_decoder(agent_queries, map_queries, map_queries, reference_points_q=fut_traj_points, reference_points_v=map_lines, pc_range=self.pc_range, map_scores=map_scores)
        co_agent_queries = torch.cat([agent_queries, extra_agent_infos], -1)
        pred_opt_traj_cls = self.traj_cls_branches[1](co_agent_queries).view(B, max_valid_query, self.num_fut_mode)
        pred_opt_traj_cls = self.log_softmax(pred_opt_traj_cls)
        pred_opt_traj = self.traj_branches[1](co_agent_queries)

        pred_opt_traj = pred_opt_traj.view(B, N, PK//self.code_size, self.code_size)

        fut_opt_traj_points = torch.cat([hist_agent_xy.unsqueeze(-2), pred_opt_traj[..., :2]], -2)
        fut_opt_traj_points = torch.cumsum(fut_opt_traj_points, -2)[:, :, 1:]


        # planner
        bs, num_agents = B, N//self.num_fut_mode
        agent_queries = self.motion_query_mlp(co_agent_queries.view(bs, num_agents, 2 * self.embed_dims * self.num_fut_mode))
        agent_reference_points = fut_opt_traj_points.view(bs, num_agents, self.num_fut_mode, 8, 2).mean(2)
        agent_centers = get_rel_pos(agent_reference_points[:, :, 0], self.pc_range)
        agent_pos = self.query_embedding(bevpos2posemb(agent_centers))

        gt_ego_lcf_feat = torch.stack(gt_ego_lcf_feat).to(agent_queries.device)
        gt_ego_fut_cmd = torch.stack(gt_ego_fut_cmd).to(agent_queries.device)

        start_of_sequence = torch.FloatTensor([
            single_img_metas['start_of_sequence'] 
            for single_img_metas in img_metas]).to(agent_queries.device)

        timestamp = torch.FloatTensor([
            single_img_metas['timestamp'] 
            for single_img_metas in img_metas]).to(agent_queries.device)

        ego_pose_inv = torch.stack([
            single_img_metas['ego_pose_inv'] 
            for single_img_metas in img_metas], 0).to(agent_queries.device)

        ego_pose = torch.stack([
            single_img_metas['ego_pose'] 
            for single_img_metas in img_metas], 0).to(agent_queries.device)

        data = dict(
            start_of_sequence = start_of_sequence,
            timestamp = timestamp,
            ego_pose_inv = ego_pose_inv,
            ego_pose = ego_pose,
        )
        fut_traj_from_velo = gt_ego_lcf_feat[:, :2].unsqueeze(1).repeat(1, self.ego_fut_steps, 1) * torch.arange(1, self.ego_fut_steps+1)[None,:, None].to(agent_queries.device) * 0.5

        self.pre_update_memory(data, fut_traj_from_velo)

        ego_query = self.query_feat_embedding.weight.repeat(bs, 1)
        ego_query = self.ego_info(ego_query, gt_ego_fut_cmd.to(ego_query.dtype)).unsqueeze(1)

        ego_pos = get_rel_pos(ego_query.new_zeros(bs, 2), self.pc_range)
        ego_pos = self.query_embedding(bevpos2posemb(ego_pos)).unsqueeze(1)
        init_ego_traj =  self.memory_traj[:, 0:1]

        hist_ego_query = self.hist_ego_mlp(self.memory_ego_embed) + self.time_embedding.weight[None]
        ego_query = self.ego_ego_decoder(
                query=ego_query,
                key=hist_ego_query,
                value=hist_ego_query,
        )
        ego_agent_query = self.ego_agent_decoder(query=ego_query,
                key=agent_queries,
                value=agent_queries,
                query_pos=ego_pos,
                key_pos=agent_pos,
                reference_points_q=init_ego_traj,
                reference_points_v=agent_reference_points)

        ego_map_query = self.ego_map_decoder(query=ego_query,
                key=map_queries,
                value=map_queries,
                query_pos=ego_pos,
                key_pos=map_pos,
                reference_points_q=init_ego_traj,
                reference_points_v=map_lines,
                map_scores=map_scores,
                )
        co_agent_query = torch.cat([ego_agent_query, ego_map_query], -1)
        outputs_ego_trajs = self.ego_fut_decoder(co_agent_query)

        outputs_ego_trajs = outputs_ego_trajs.reshape(outputs_ego_trajs.shape[0], 
                                                      self.ego_fut_mode, self.ego_fut_steps, 2)

        self.post_update_memory(data, torch.cumsum(outputs_ego_trajs[gt_ego_fut_cmd==1], 1)[:, None], co_agent_query)
        
        ego_trajs = torch.cumsum(outputs_ego_trajs[gt_ego_fut_cmd==1], 1)
        ego_trajs = torch.cat([torch.zeros_like(ego_trajs[:,:1]), ego_trajs], 1)
        ego_trajs = torch.cat([ego_trajs, torch.zeros_like(ego_trajs[..., :1])], -1)
        ego_trajs_in_global = transform_reference_points(ego_trajs, data['ego_pose'], reverse=False)[..., :2]

        fut_trajs_in_global = torch.cat([fut_opt_traj_points, torch.zeros_like(fut_opt_traj_points[..., :1])], -1)
        fut_trajs_in_global = transform_reference_points(fut_trajs_in_global, data['ego_pose'], reverse=False)[..., :2]

        return dict(
            pred_trajs=[
                dict(
                    pred_traj=pred_traj.view(B, N//self.num_fut_mode, self.num_fut_mode,  PK//self.code_size, self.code_size),
                    pred_traj_cls=pred_traj_cls,
                    valid_length=valid_length,
                ),
                dict(
                    pred_traj=pred_opt_traj.view(B, N//self.num_fut_mode, self.num_fut_mode, PK//self.code_size, self.code_size),
                    pred_traj_cls=pred_opt_traj_cls,
                    valid_length=valid_length,
            )],
            fut_traj_from_velo = fut_traj_from_velo,
            fut_trajs_in_global = fut_trajs_in_global,
            pred_abs_trajs2 = fut_opt_traj_points,
            obj_idxes = agent_instances.obj_idxes.clone(),
            agent_scores = agent_instances.scores.clone(),
            ego_fut_preds=outputs_ego_trajs,
            ego_trajs_in_global = ego_trajs_in_global,
            )
        
    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_agent_fut_traj=None,
             gt_agent_fut_traj_mask=None,
             gt_ego_fut_trajs=None,
             gt_ego_fut_cmd=None,
             gt_ego_fut_masks=None,
             preds_dicts=None,
             preds_map_dicts=None,
             matched_gt_idxes=None,
             img_metas=None,
            ):

        loss_dict = dict()
        gt_agent_fut_traj_list = []
        gt_agent_fut_traj_mask_list = []
        B = len(gt_agent_fut_traj)
        pred_trajs = preds_dicts['pred_trajs']
        valid_length = pred_trajs[0]['valid_length']

        for i in range(B):
            index = matched_gt_idxes[i][:valid_length[i]]
            if valid_length[i]>0:
                gt_agent_fut_traj_list.append(gt_agent_fut_traj[i][:valid_length[i]][index])
                gt_agent_fut_traj_mask_list.append(gt_agent_fut_traj_mask[i][:valid_length[i]][index])

        gt_agent_fut_traj = torch.cat(gt_agent_fut_traj_list)
        gt_agent_fut_traj_mask = torch.cat(gt_agent_fut_traj_mask_list).sum(-1) > 0

        for lld, single_preds in  enumerate(pred_trajs):
            pred_traj = single_preds['pred_traj']
            pred_traj_cls = single_preds['pred_traj_cls']
            pred_agent_fut_traj_list = []
            pred_agent_fut_traj_cls_list = []

            for i in range(B):
                if valid_length[i]>0:
                    pred_agent_fut_traj_list.append(pred_traj[i][:valid_length[i]])
                    pred_agent_fut_traj_cls_list.append(pred_traj_cls[i][:valid_length[i]])

            pred_traj = torch.cat(pred_agent_fut_traj_list)
            pred_traj_cls = torch.cat(pred_agent_fut_traj_cls_list)
            loss_traj, l_class, l_reg, l_minade, l_minfde, l_mr = self.loss_traj(pred_traj_cls, pred_traj, gt_agent_fut_traj, gt_agent_fut_traj_mask)
            loss_dict.update({
                    f'loss_traj.d{lld}': loss_traj,
                    f'l_class.d{lld}': l_class,
                    f'l_reg.d{lld}': l_reg,
                    f'l_minade.d{lld}': l_minade,
                    f'l_minfde.d{lld}': l_minfde,
                    f'l_mr.d{lld}': l_mr,
                }
            )

        ego_fut_preds = preds_dicts['ego_fut_preds']
        map_lines = preds_map_dicts['lines']
        B, NMQ, K2 = map_lines.shape
        map_lines = map_lines.reshape(B, NMQ, K2//2, 2)
        map_scores = preds_map_dicts['scores']

        agent_fut_preds = preds_dicts['pred_abs_trajs2'].reshape(B, -1, self.num_fut_mode, 8, 2)[..., :self.ego_fut_steps, :2]
        agent_score_preds = preds_dicts['agent_scores']
        agent_fut_cls_preds = preds_dicts['pred_trajs'][-1]['pred_traj_cls']
        gt_ego_fut_trajs = torch.stack(gt_ego_fut_trajs)
        gt_ego_fut_cmd = torch.stack(gt_ego_fut_cmd)
        gt_ego_fut_masks = torch.stack(gt_ego_fut_masks)
        gt_ego_fut_trajs = torch.cat([gt_ego_fut_trajs[:,:1], (gt_ego_fut_trajs[:,1:] - gt_ego_fut_trajs[:,:-1])], 1)
        gt_ego_fut_trajs = gt_ego_fut_trajs.unsqueeze(1).repeat(1, self.ego_fut_mode, 1, 1)

        loss_plan_l1_weight = gt_ego_fut_cmd[..., None, None] * gt_ego_fut_masks[:, None, :, None]
        loss_plan_l1_weight = loss_plan_l1_weight.repeat(1, 1, 1, 2)
        
        loss_plan_l1 = self.loss_plan_reg(
            ego_fut_preds,
            gt_ego_fut_trajs,
            loss_plan_l1_weight
        )

        loss_plan_bound = self.loss_plan_bound(
            ego_fut_preds[gt_ego_fut_cmd==1],
            map_lines,
            map_scores,
            weight=gt_ego_fut_masks
        )

        loss_plan_col = self.loss_plan_col(
            ego_fut_preds[gt_ego_fut_cmd==1],
            agent_fut_preds,
            agent_score_preds.squeeze(-1),
            agent_fut_cls_preds,
            weight=gt_ego_fut_masks[:, :, None].repeat(1, 1, 2)
        )

        loss_plan_dir = self.loss_plan_dir(
            ego_fut_preds[gt_ego_fut_cmd==1],
            map_lines,
            map_scores,
            weight=gt_ego_fut_masks
        )

        loss_plan_l1 = torch.nan_to_num(loss_plan_l1)
        loss_plan_bound = torch.nan_to_num(loss_plan_bound)
        loss_plan_col = torch.nan_to_num(loss_plan_col)
        loss_plan_dir = torch.nan_to_num(loss_plan_dir)
        
        loss_dict['loss_plan_reg'] = loss_plan_l1
        loss_dict['loss_plan_bound'] = loss_plan_bound
        loss_dict['loss_plan_col'] = loss_plan_col
        loss_dict['loss_plan_dir'] = loss_plan_dir

        return loss_dict


    @force_fp32(apply_to=('preds_dicts'))
    def get_motion(self, preds_dicts, img_metas,  rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        num_samples = len(img_metas)

        # fut_traj_points = preds_dicts['pred_abs_trajs'].view(num_samples, -1, self.num_fut_mode,  self.fut_steps, self.code_size)
        # # fut_traj_index = preds_dicts['pred_trajs'][0]['pred_traj_cls'].softmax(-1).argmax(-1)
        # scores, fut_traj_index = torch.max(preds_dicts['pred_trajs'][0]['pred_traj_cls'].softmax(-1), dim=-1)
        # inds_rep = fut_traj_index.repeat(
        #     self.fut_steps, self.code_size, 1, 1).permute(2, 3, 0, 1)
        # fut_traj_points = fut_traj_points.gather(2, inds_rep.unsqueeze(2)).squeeze(dim=2)

        fut_traj_points2 = preds_dicts['pred_abs_trajs2'].view(num_samples, -1, self.num_fut_mode,  self.fut_steps, self.code_size)
        fut_trajs_in_global = preds_dicts['fut_trajs_in_global'].view(num_samples, -1, self.num_fut_mode,  self.fut_steps, self.code_size)
        # fut_traj_index = preds_dicts['pred_trajs'][0]['pred_traj_cls'].softmax(-1).argmax(-1)
        # scores, fut_traj_index = torch.max(preds_dicts['pred_trajs'][1]['pred_traj_cls'].softmax(-1), dim=-1)
        # inds_rep = fut_traj_index.repeat(
        #    self.fut_steps, self.code_size, 1, 1).permute(2, 3, 0, 1)
        # fut_traj_points2 = fut_traj_points2.gather(2, inds_rep.unsqueeze(2)).squeeze(dim=2)

        ret_list = [] 
        for i in range(num_samples):
            ret_list.append(
                dict(
                    # pred_agent_fut_trajs = fut_traj_points[i].cpu().numpy(),
                    fut_trajs_in_global = fut_trajs_in_global[i].cpu().numpy(),
                    pred_agent_fut_trajs2 = fut_traj_points2[i].cpu().numpy(),
                    pred_traj_cls = preds_dicts['pred_trajs'][1]['pred_traj_cls'][i].softmax(-1).cpu().numpy(),
                    pred_traj = preds_dicts['pred_trajs'][1]['pred_traj'][i].cpu().numpy(),
                    obj_idxes = preds_dicts['obj_idxes'][i].cpu().numpy()
                    )
                )
        return ret_list


    @force_fp32(apply_to=('preds_dicts'))
    def get_traj(self, preds_dicts, img_metas,  rescale=False, gt_ego_fut_trajs=None, gt_ego_fut_cmd=None, gt_ego_fut_masks=None, gt_fut_segmentations=None, vad_ego_fut_trajs=None):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        pred_ego_fut_trajs = preds_dicts['ego_fut_preds']

        gt_ego_fut_trajs = torch.stack(gt_ego_fut_trajs).to(pred_ego_fut_trajs.device)
        gt_ego_fut_cmd = torch.stack(gt_ego_fut_cmd).to(pred_ego_fut_trajs.device)
        gt_ego_fut_masks = torch.stack(gt_ego_fut_masks).to(pred_ego_fut_trajs.device)

        pred_ego_fut_trajs = torch.cumsum(pred_ego_fut_trajs[gt_ego_fut_cmd==1], 1)
        # pred_ego_fut_trajs = vad_ego_fut_trajs[0][None]

        metric_dict = {
            'plan_L2_1s':0,
            'plan_L2_2s':0,
            'plan_L2_3s':0,
            'plan_obj_col_1s':0,
            'plan_obj_col_2s':0,
            'plan_obj_col_3s':0,
            'plan_obj_box_col_1s':0,
            'plan_obj_box_col_2s':0,
            'plan_obj_box_col_3s':0,
            'l2_dist': 0,
        }
        

        fut_valid_flag = gt_ego_fut_masks.all()
        future_second = 3
        metric_dict['fut_valid_flag'] = fut_valid_flag.cpu().item()
        for i in range(future_second):
            if fut_valid_flag:
                cur_time = (i+1)*2
                traj_L2 = self.planning_metric.compute_L2(
                    pred_ego_fut_trajs[0, :cur_time].detach().to(gt_ego_fut_trajs.device),
                    gt_ego_fut_trajs[0, :cur_time]
                )

                obj_coll, obj_box_coll = self.planning_metric.evaluate_coll(
                    pred_ego_fut_trajs[:, :cur_time].detach().to(gt_ego_fut_trajs.device),
                    gt_ego_fut_trajs[:, :cur_time],
                    gt_fut_segmentations,
                    index = [each['index'] for each in img_metas]
                )
                metric_dict['plan_L2_{}s'.format(i+1)] = traj_L2
                metric_dict['plan_obj_col_{}s'.format(i+1)] = obj_coll.mean().item()
                metric_dict['plan_obj_box_col_{}s'.format(i+1)] = obj_box_coll.max().item()
        l2_dist = (pred_ego_fut_trajs-gt_ego_fut_trajs).norm(dim=-1) * gt_ego_fut_masks[:, None]

        l2_dist[gt_ego_fut_masks[:, None]==0] = -1
        metric_dict['l2_dist'] = l2_dist[0].cpu()
        ret_list = []
        num_samples = len(pred_ego_fut_trajs)
        assert num_samples == 1
        
        index_w_scene = img_metas[0]['scene_name'] + '-' + str(img_metas[0]['index'])

        for i in range(num_samples):
            ret_list.append(
                dict(
                    pred_ego_fut_trajs = pred_ego_fut_trajs[i].cpu(),
                    gt_ego_fut_trajs = gt_ego_fut_trajs[i].cpu(),
                    metric_dict = metric_dict,
                    l2_dist=l2_dist[i].cpu(),
                    index_w_scene = index_w_scene,
                    ego_trajs_in_global = preds_dicts['ego_trajs_in_global'][i].cpu(),
                    gt_ego_fut_cmd = gt_ego_fut_cmd[i].cpu(),
                    )
                )
        return ret_list

class MLN(nn.Module):
    ''' 
    Args:
        c_dim (int): dimension of latent code c
        f_dim (int): feature dimension
    '''

    def __init__(self, c_dim, f_dim=256, use_ln=True):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.use_ln = use_ln

        self.reduce = nn.Sequential(
            nn.Linear(c_dim, f_dim),
            nn.ReLU(),
        )
        self.gamma = nn.Linear(f_dim, f_dim)
        self.beta = nn.Linear(f_dim, f_dim)
        if self.use_ln:
            self.ln = nn.LayerNorm(f_dim, elementwise_affine=False)
        self.init_weight()

    def init_weight(self):
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, c):
        if self.use_ln:
            x = self.ln(x)
        c = self.reduce(c)
        gamma = self.gamma(c)
        beta = self.beta(c)
        out = gamma * x + beta

        return out