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
class MotionHead(BaseModule):
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


        super(MotionHead, self).__init__()

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


    def forward(self, agent_instances, preds_map_dicts, img_metas=None):
        
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
        
        map_queries = preds_map_dicts[self.map_layer_index]['queries'].clone()
        map_lines = preds_map_dicts[self.map_layer_index]['lines'].clone()
        map_scores = preds_map_dicts[self.map_layer_index]['scores'].clone()
        B, NMQ, K2 = map_lines.shape
        map_lines = map_lines.reshape(B, NMQ, K2//2, 2)
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
            pred_abs_trajs = fut_traj_points,
            pred_abs_trajs2 = fut_opt_traj_points,
            obj_idxes = agent_instances.obj_idxes.clone(),
            motion_queries = agent_queries,
            agent_logits = agent_instances.logits.clone()
            )
        
    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_agent_fut_traj,
             gt_agent_fut_traj_mask,
             preds_dicts,
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

        # from IPython import embed
        # embed()
        # exit()

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
        return loss_dict


    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas,  rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        num_samples = len(img_metas)
        fut_traj_points = preds_dicts['pred_abs_trajs'].view(num_samples, -1, self.num_fut_mode,  self.fut_steps, self.code_size)
        # fut_traj_index = preds_dicts['pred_trajs'][0]['pred_traj_cls'].softmax(-1).argmax(-1)
        scores, fut_traj_index = torch.max(preds_dicts['pred_trajs'][0]['pred_traj_cls'].softmax(-1), dim=-1)
        inds_rep = fut_traj_index.repeat(
            self.fut_steps, self.code_size, 1, 1).permute(2, 3, 0, 1)
        fut_traj_points = fut_traj_points.gather(2, inds_rep.unsqueeze(2)).squeeze(dim=2)

        fut_traj_points2 = preds_dicts['pred_abs_trajs2'].view(num_samples, -1, self.num_fut_mode,  self.fut_steps, self.code_size)
        # fut_traj_index = preds_dicts['pred_trajs'][0]['pred_traj_cls'].softmax(-1).argmax(-1)
        scores, fut_traj_index = torch.max(preds_dicts['pred_trajs'][1]['pred_traj_cls'].softmax(-1), dim=-1)
        inds_rep = fut_traj_index.repeat(
            self.fut_steps, self.code_size, 1, 1).permute(2, 3, 0, 1)
        fut_traj_points2 = fut_traj_points2.gather(2, inds_rep.unsqueeze(2)).squeeze(dim=2)

        ret_list = [] 
        for i in range(num_samples):
            ret_list.append(
                dict(
                    pred_agent_fut_trajs = fut_traj_points[i].cpu().numpy(),
                    pred_agent_fut_trajs2 = fut_traj_points2[i].cpu().numpy(),
                    obj_idxes = preds_dicts['obj_idxes'][i].cpu().numpy()
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