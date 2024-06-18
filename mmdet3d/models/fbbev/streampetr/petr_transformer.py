import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init, build_norm_layer
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence,
                                         build_attention,
                                         build_feedforward_network)
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction
from mmcv.runner.base_module import BaseModule
from mmcv.cnn.bricks.registry import (ATTENTION,TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmdet.models.utils.builder import TRANSFORMER
from .streampetr_utils import  pos2posemb3d, bevpos2posemb
from mmdet.models.utils.transformer import inverse_sigmoid
from mmcv.utils import deprecated_api_warning, ConfigDict
import warnings
import copy
from torch.nn import ModuleList
import torch.utils.checkpoint as cp
from mmcv.runner import force_fp32
from torch.cuda.amp import autocast

# Disable warnings
warnings.filterwarnings("ignore")

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


@TRANSFORMER.register_module()
class Detr3DTransformer(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 decoder=None,
                 **kwargs):
        super(Detr3DTransformer, self).__init__(**kwargs)
        self.decoder = build_transformer_layer_sequence(decoder)

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def forward(self,
                query,
                query_pos,
                feat_flatten,
                spatial_flatten,
                level_start_index, 
                temp_memory, 
                temp_pos,
                attn_masks,
                reference_points, 
                pc_range, 
                data, 
                img_metas,
                temp_reference_points=None,
                reg_branches=None,
                query_embedding=None,
                return_intermediate_pts=False,
                cam_params=None,
                debug_info=None,
                ):
        """Forward function for `Detr3DTransformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, embed_dims, h, w].
            query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            mlvl_pos_embeds (list(Tensor)): The positional encoding
                of feats from different level, has the shape
                 [bs, embed_dims, h, w].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when
                `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """
        lidar2img = None #  data['lidar2img']
        
        inter_states = self.decoder(
            query=query,
            query_pos=query_pos,
            mlvl_feats=feat_flatten,
            temp_memory=temp_memory, 
            temp_pos=temp_pos,
            reference_points=reference_points,
            spatial_flatten=spatial_flatten,
            level_start_index=level_start_index,
            pc_range=pc_range, 
            lidar2img=lidar2img, 
            img_metas=img_metas,
            attn_masks=attn_masks,
            reg_branches=reg_branches,
            query_embedding=query_embedding,
            return_intermediate_pts=return_intermediate_pts,
            cam_params=cam_params,
            debug_info=debug_info,
            temp_reference_points=temp_reference_points,
            )

        return inter_states

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class Detr3DTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, embed_dims, *args,  predict_refine=True, **kwargs):
        self.predict_refine =predict_refine
        super(Detr3DTransformerDecoder, self).__init__(*args, **kwargs)

    def forward(self,
                query,
                query_pos,
                mlvl_feats,
                temp_memory, 
                temp_pos,
                reference_points,
                spatial_flatten,
                level_start_index,
                pc_range, 
                lidar2img, 
                img_metas,
                attn_masks,
                temp_reference_points=None,
                reg_branches=None,
                query_embedding=None,
                return_intermediate_pts=False,
                cam_params=None,
                debug_info=None,
               
                ):
        """Forward function for `Detr3DTransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        intermediate = []
        intermediate_reference_points = []
        ori_reference_points = reference_points.clone()
        for lid, layer in enumerate(self.layers):
            
            query = layer(
                query,
                query_pos,
                mlvl_feats,
                temp_memory, 
                temp_pos,
                reference_points,
                spatial_flatten,
                level_start_index,
                pc_range, 
                lidar2img, 
                img_metas,
                attn_masks,
                temp_reference_points=temp_reference_points,
                cam_params=cam_params,
                debug_info=debug_info,
                )
            if reg_branches is not None:
                ref_shape = reference_points.shape

                if len(ref_shape) == 3: # Detection
                    reg_points = reg_branches[lid](query)[..., :3].reshape(*ref_shape)
                elif len(ref_shape) == 4: # Map
                    reg_points = reg_branches[lid](query).reshape(*ref_shape)
                if self.predict_refine:
                    new_reference_points = reg_points + inverse_sigmoid(reference_points)
                else:
                    if len(ref_shape) == 3: # Detection predicts the offset from the initial reference_points
                        new_reference_points = reg_points + inverse_sigmoid(ori_reference_points)
                    elif len(ref_shape) == 4: # Map predcits absolute reference points
                        new_reference_points = reg_points
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.clone().detach()
                intermediate_reference_points.append(new_reference_points) # Look twice from DINO
                if lid < len(self.layers)-1 and query_embedding is not None:
                    if len(ref_shape) == 3: # Detection
                        query_pos = query_embedding(pos2posemb3d(reference_points))
                    elif len(ref_shape) == 4: # Map
                        query_pos = query_embedding(bevpos2posemb(reference_points, 32).flatten(-2, -1))

            intermediate.append(query)
        if return_intermediate_pts:
            return  torch.stack(intermediate),  torch.stack(intermediate_reference_points)
        return torch.stack(intermediate)

@TRANSFORMER_LAYER.register_module()
class Detr3DTemporalDecoderLayer(BaseModule):
    """Base `TransformerLayer` for vision transformer.

    It can be built from `mmcv.ConfigDict` and support more flexible
    customization, for example, using any number of `FFN or LN ` and
    use different kinds of `attention` by specifying a list of `ConfigDict`
    named `attn_cfgs`. It is worth mentioning that it supports `prenorm`
    when you specifying `norm` as the first element of `operation_order`.
    More details about the `prenorm`: `On Layer Normalization in the
    Transformer Architecture <https://arxiv.org/abs/2002.04745>`_ .

    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
            Configs for `self_attention` or `cross_attention` modules,
            The order of the configs in the list should be consistent with
            corresponding attentions in operation_order.
            If it is a dict, all of the attention modules in operation_order
            will be built with this config. Default: None.
        ffn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
            Configs for FFN, The order of the configs in the list should be
            consistent with corresponding ffn in operation_order.
            If it is a dict, all of the attention modules in operation_order
            will be built with this config.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Support `prenorm` when you specifying first element as `norm`.
            Default：None.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): Key, Query and Value are shape
            of (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
    """

    def __init__(self,
                 attn_cfgs=None,
                 ffn_cfgs=dict(
                     type='FFN',
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True),
                 ),
                 operation_order=None,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None,
                 batch_first=False,
                 with_cp=True,
                 **kwargs):
        super().__init__(init_cfg)

        self.batch_first = batch_first

        assert set(operation_order) & {
            'self_attn', 'norm', 'ffn', 'cross_attn'} == \
            set(operation_order), f'The operation_order of' \
            f' {self.__class__.__name__} should ' \
            f'contains all four operation type ' \
            f"{['self_attn', 'norm', 'ffn', 'cross_attn']}"

        num_attn = operation_order.count('self_attn') + operation_order.count(
            'cross_attn')
        if isinstance(attn_cfgs, dict):
            attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]
        else:
            assert num_attn == len(attn_cfgs), f'The length ' \
                f'of attn_cfg {num_attn} is ' \
                f'not consistent with the number of attention' \
                f'in operation_order {operation_order}.'

        self.num_attn = num_attn
        self.operation_order = operation_order
        self.norm_cfg = norm_cfg
        self.pre_norm = operation_order[0] == 'norm'
        self.attentions = ModuleList()

        index = 0
        for operation_name in operation_order:
            if operation_name in ['self_attn', 'cross_attn']:
                if 'batch_first' in attn_cfgs[index]:
                    assert self.batch_first == attn_cfgs[index]['batch_first']
                else:
                    attn_cfgs[index]['batch_first'] = self.batch_first
                attention = build_attention(attn_cfgs[index])
                # Some custom attentions used as `self_attn`
                # or `cross_attn` can have different behavior.
                attention.operation_name = operation_name
                self.attentions.append(attention)
                index += 1

        self.embed_dims = self.attentions[0].embed_dims

        self.ffns = ModuleList()
        num_ffns = operation_order.count('ffn')
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = ConfigDict(ffn_cfgs)
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = [copy.deepcopy(ffn_cfgs) for _ in range(num_ffns)]
        assert len(ffn_cfgs) == num_ffns
        for ffn_index in range(num_ffns):
            if 'embed_dims' not in ffn_cfgs[ffn_index]:
                ffn_cfgs[ffn_index]['embed_dims'] = self.embed_dims
            else:
                assert ffn_cfgs[ffn_index]['embed_dims'] == self.embed_dims
            self.ffns.append(
                build_feedforward_network(ffn_cfgs[ffn_index],
                                          dict(type='FFN')))

        self.norms = ModuleList()
        num_norms = operation_order.count('norm')
        for _ in range(num_norms):
            self.norms.append(build_norm_layer(norm_cfg, self.embed_dims)[1])

        self.use_checkpoint = with_cp

    def _forward(self,
                query,
                query_pos,
                mlvl_feats,
                temp_memory, 
                temp_pos,
                reference_points,
                spatial_flatten,
                level_start_index,
                pc_range, 
                lidar2img, 
                img_metas,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                temp_reference_points=None,
                cam_params=None,
                debug_info=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                        f'attn_masks {len(attn_masks)} must be equal ' \
                        f'to the number of attention in ' \
                        f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            if layer == 'self_attn':
                if temp_memory is not None:
                    temp_key = temp_value = torch.cat([query, temp_memory], dim=1)
                    if query_pos is not None and temp_pos is not None: 
                        temp_pos = torch.cat([query_pos, temp_pos], dim=1)
                    temp_reference_points = torch.cat([reference_points, temp_reference_points], dim=1)
                else:
                    temp_key = temp_value = query
                    temp_pos = query_pos
                    temp_reference_points = reference_points
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=temp_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    reference_points=reference_points,
                    temp_reference_points=temp_reference_points,
                    pc_range=pc_range,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    query_pos,
                    mlvl_feats,
                    reference_points,
                    spatial_flatten,
                    level_start_index,
                    pc_range, 
                    lidar2img, 
                    img_metas,
                    cam_params=cam_params,
                    debug_info=debug_info,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query

    def forward(self, 
                query,
                query_pos,
                mlvl_feats,
                temp_memory, 
                temp_pos,
                reference_points,
                spatial_flatten,
                level_start_index,
                pc_range, 
                lidar2img, 
                img_metas,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                temp_reference_points=None,
                cam_params=None,
                debug_info=None,
                ):
        """Forward function for `TransformerCoder`.
        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if self.use_checkpoint and self.training:
            x = cp.checkpoint(
                self._forward, 
                query,
                query_pos,
                mlvl_feats,
                temp_memory, 
                temp_pos,
                reference_points,
                spatial_flatten,
                level_start_index,
                pc_range, 
                lidar2img, 
                img_metas,
                attn_masks,
                query_key_padding_mask,
                key_padding_mask,
                temp_reference_points,
                cam_params,
                debug_info
                )
        else:
            x = self._forward(
            query,
            query_pos,
            mlvl_feats,
            temp_memory, 
            temp_pos,
            reference_points,
            spatial_flatten,
            level_start_index,
            pc_range, 
            lidar2img, 
            img_metas,
            attn_masks,
            query_key_padding_mask,
            key_padding_mask,
            temp_reference_points=temp_reference_points,
            cam_params=cam_params,
            debug_info=debug_info,
        )
        return x


@ATTENTION.register_module()
class DeformableFeatureAggregationCuda(BaseModule):
    def __init__(
            self,
            embed_dims=256,
            num_groups=8,
            num_levels=4,
            num_cams=6,
            dropout=0.1,
            num_pts=13,
            num_anchor_pts=1,
            im2col_step=64,
            batch_first=True,
            code_size=3,
            bias=1.,
            ):
        super(DeformableFeatureAggregationCuda, self).__init__()
        self.embed_dims = embed_dims
        self.num_groups = num_groups
        self.num_anchor_pts = num_anchor_pts
        self.group_dims = (self.embed_dims // self.num_groups)
        self.num_levels = num_levels
        self.num_cams = num_cams
        self.num_pts = num_pts
        self.code_size = code_size
        self.weights_fc = nn.Linear(self.embed_dims, self.num_groups * self.num_levels * num_pts * self.num_anchor_pts)
        self.output_proj = nn.Linear(self.embed_dims, self.embed_dims)
        self.learnable_fc = nn.Linear(self.embed_dims, self.num_anchor_pts * num_pts * code_size)
        # self.cam_embed = nn.Sequential(
        #     nn.Linear(12, self.embed_dims // 2),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.embed_dims // 2, self.embed_dims),
        #     nn.ReLU(inplace=True),
        #     nn.LayerNorm(self.embed_dims),
        # )
        self.drop = nn.Dropout(dropout)
        self.im2col_step = im2col_step
        self.bias = bias

    def init_weight(self):
        constant_init(self.weights_fc, val=0.0, bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)
        nn.init.uniform_(self.learnable_fc.bias.data, -self.bias, self.bias)    

    @force_fp32()
    def forward(self, instance_feature, query_pos, feat_flatten, reference_points, spatial_flatten, level_start_index, pc_range, lidar2img_mat, img_metas, cam_params=None, debug_info=None):
        bs, num_query = reference_points.shape[:2]
        reference_points = get_ego_pos(reference_points, pc_range)
        if reference_points.dim()==3 and self.num_anchor_pts==1:
            key_points = reference_points.unsqueeze(-2) + self.learnable_fc(instance_feature).reshape(bs, num_query, -1, self.code_size)
        elif reference_points.dim()==4 and self.num_anchor_pts==reference_points.size(2): # one query has more than 1 reference points
            key_points = reference_points.unsqueeze(-2) + self.learnable_fc(instance_feature).reshape(bs, num_query, self.num_anchor_pts, -1, self.code_size)
           
            key_points = key_points.reshape(bs, num_query, self.num_anchor_pts * self.num_pts, self.code_size)
        key_points = get_rel_pos(key_points, pc_range)
        weights = self._get_weights(instance_feature, query_pos, lidar2img_mat)

        features = self.feature_sampling(feat_flatten, spatial_flatten, level_start_index, key_points, weights, lidar2img_mat, img_metas)

        output = self.output_proj(features)
        output = self.drop(output) + instance_feature
        return output

    def _get_weights(self, instance_feature, anchor_embed, lidar2img_mat):
        bs, num_query = instance_feature.shape[:2]
        # lidar2img = lidar2img_mat[..., :3, :].flatten(-2)
        # cam_embed = self.cam_embed(lidar2img) # B, N, C
        if anchor_embed is not None:
            feat_pos = (instance_feature + anchor_embed) # .unsqueeze(2)  # + cam_embed.unsqueeze(1)
        else:
            feat_pos = instance_feature

        if self.num_anchor_pts==1:
            weights = self.weights_fc(feat_pos).reshape(bs, num_query, self.num_groups, -1).softmax(dim=-1)
            weights = weights.reshape(bs, num_query, self.num_groups, self.num_levels, self.num_pts).contiguous()
        else:
            weights = self.weights_fc(feat_pos).reshape(bs, num_query, self.num_groups, self.num_anchor_pts, -1).softmax(dim=-1) / self.num_anchor_pts
            weights = weights.reshape(bs, num_query, self.num_groups, self.num_anchor_pts, self.num_levels, self.num_pts)
            weights = weights.permute(0, 1, 2, 4, 3, 5).flatten(-2).contiguous()

        return weights

    def feature_sampling(self, feat_flatten, spatial_flatten, level_start_index, key_points, weights, lidar2img_mat, img_metas):
        bs, num_query, _ = key_points.shape[:3]

        # pts_extand = torch.cat([key_points, torch.ones_like(key_points[..., :1])], dim=-1)
        # points_2d = torch.matmul(lidar2img_mat[:, :, None, None], pts_extand[:, None, ..., None]).squeeze(-1)

        # points_2d = points_2d[..., :2] / torch.clamp(points_2d[..., 2:3], min=1e-5)
        # points_2d[..., 0:1] = points_2d[..., 0:1] / img_metas[0]['pad_shape'][0][1]
        # points_2d[..., 1:2] = points_2d[..., 1:2] / img_metas[0]['pad_shape'][0][0]

        # points_2d = points_2d.flatten(end_dim=1) #[b*6, 900, 13, 2]
        # points_2d = points_2d[:, :, None, None, :, :].repeat(1, 1, self.num_groups, self.num_levels, 1, 1)

        points_2d = key_points[..., :2]
        points_2d = points_2d[:, :, None, None, :, :].repeat(1, 1, self.num_groups, self.num_levels, 1, 1)

        bn, num_value, _ = feat_flatten.size()
        feat_flatten = feat_flatten.reshape(bn, num_value, self.num_groups, -1)
        # attention_weights = weights * mask
        with autocast(enabled=False):
            output = MultiScaleDeformableAttnFunction.apply(
                feat_flatten, spatial_flatten, level_start_index, points_2d,
                weights, self.im2col_step)
        
        output = output.reshape(bs, num_query, -1)

        return output


@ATTENTION.register_module()
class DeformableFeatureAggregationCuda_v2(BaseModule):
    def __init__(
            self,
            embed_dims=256,
            num_groups=8,
            num_levels=4,
            num_cams=6,
            dropout=0.1,
            num_pts=13,
            num_anchor_pts=1,
            im2col_step=64,
            batch_first=True,
           
            bias=1.,
            ):
        super(DeformableFeatureAggregationCuda_v2, self).__init__()
        self.embed_dims = embed_dims
        self.num_groups = num_groups
        self.num_anchor_pts = num_anchor_pts
        self.group_dims = (self.embed_dims // self.num_groups)
        self.num_levels = num_levels
        self.num_cams = num_cams
        self.num_pts = num_pts
        self.weights_fc = nn.Linear(self.embed_dims, self.num_groups * self.num_levels * num_pts * self.num_anchor_pts)
        self.output_proj = nn.Linear(self.embed_dims, self.embed_dims)
        self.learnable_fc = nn.Linear(self.embed_dims, self.num_anchor_pts * num_pts * 3)
        # self.cam_embed = nn.Sequential(
        #     nn.Linear(12, self.embed_dims // 2),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.embed_dims // 2, self.embed_dims),
        #     nn.ReLU(inplace=True),
        #     nn.LayerNorm(self.embed_dims),
        # )
        self.drop = nn.Dropout(dropout)
        self.im2col_step = im2col_step
        self.bias = bias

    def init_weight(self):
        constant_init(self.weights_fc, val=0.0, bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)
        nn.init.uniform_(self.learnable_fc.bias.data, -self.bias, self.bias)    

    @force_fp32()
    def forward(self, instance_feature, query_pos, feat_flatten, reference_points, spatial_flatten, level_start_index, pc_range, lidar2img_mat, img_metas, cam_params=None, debug_info=None):
        bs, num_query = reference_points.shape[:2]
        reference_points = get_ego_pos(reference_points, pc_range)
        if reference_points.dim()==3 and self.num_anchor_pts==1:
            key_points = reference_points.unsqueeze(-2) + self.learnable_fc(instance_feature+query_pos).reshape(bs, num_query, -1, 3)
        elif reference_points.dim()==4 and self.num_anchor_pts==reference_points.size(2): # one query has more than 1 reference points
            key_points = reference_points.unsqueeze(-2) + self.learnable_fc(instance_feature+query_pos).reshape(bs, num_query, self.num_anchor_pts, -1, 3)
            key_points = key_points.reshape(bs, num_query, self.num_anchor_pts * self.num_pts, 3)
        key_points = get_rel_pos(key_points, pc_range)
        weights = self._get_weights(instance_feature, query_pos, lidar2img_mat)

        features = self.feature_sampling(feat_flatten, spatial_flatten, level_start_index, key_points, weights, lidar2img_mat, img_metas)

        output = self.output_proj(features)
        output = self.drop(output) + instance_feature
        return output

    def _get_weights(self, instance_feature, anchor_embed, lidar2img_mat):
        bs, num_query = instance_feature.shape[:2]
        # lidar2img = lidar2img_mat[..., :3, :].flatten(-2)
        # cam_embed = self.cam_embed(lidar2img) # B, N, C
        feat_pos = instance_feature + anchor_embed # .unsqueeze(2)  # + cam_embed.unsqueeze(1)
        if self.num_anchor_pts==1:
            weights = self.weights_fc(feat_pos).reshape(bs, num_query, self.num_groups, -1).softmax(dim=-1)
            weights = weights.reshape(bs, num_query, self.num_groups, self.num_levels, self.num_pts).contiguous()
        else:
            weights = self.weights_fc(feat_pos).reshape(bs, num_query, self.num_groups, self.num_anchor_pts, -1).softmax(dim=-1) / self.num_anchor_pts
            weights = weights.reshape(bs, num_query, self.num_groups, self.num_anchor_pts, self.num_levels, self.num_pts)
            weights = weights.permute(0, 1, 2, 4, 3, 5).flatten(-2).contiguous()

        return weights

    def feature_sampling(self, feat_flatten, spatial_flatten, level_start_index, key_points, weights, lidar2img_mat, img_metas):
        bs, num_query, _ = key_points.shape[:3]

        # pts_extand = torch.cat([key_points, torch.ones_like(key_points[..., :1])], dim=-1)
        # points_2d = torch.matmul(lidar2img_mat[:, :, None, None], pts_extand[:, None, ..., None]).squeeze(-1)

        # points_2d = points_2d[..., :2] / torch.clamp(points_2d[..., 2:3], min=1e-5)
        # points_2d[..., 0:1] = points_2d[..., 0:1] / img_metas[0]['pad_shape'][0][1]
        # points_2d[..., 1:2] = points_2d[..., 1:2] / img_metas[0]['pad_shape'][0][0]

        # points_2d = points_2d.flatten(end_dim=1) #[b*6, 900, 13, 2]
        # points_2d = points_2d[:, :, None, None, :, :].repeat(1, 1, self.num_groups, self.num_levels, 1, 1)

        points_2d = key_points[..., :2]
        points_2d = points_2d[:, :, None, None, :, :].repeat(1, 1, self.num_groups, self.num_levels, 1, 1)

        bn, num_value, _ = feat_flatten.size()
        feat_flatten = feat_flatten.reshape(bn, num_value, self.num_groups, -1)
        # attention_weights = weights * mask
        with autocast(enabled=False):
            output = MultiScaleDeformableAttnFunction.apply(
                feat_flatten, spatial_flatten, level_start_index, points_2d,
                weights, self.im2col_step)
        
        output = output.reshape(bs, num_query, -1)

        return output




@ATTENTION.register_module()
class MVDeformableFeatureAggregationCuda(BaseModule):
    def __init__(
            self,
            embed_dims=256,
            num_groups=8,
            num_levels=4,
            num_cams=6,
            dropout=0.1,
            num_pts=13,
            im2col_step=64,
            batch_first=True,
            bias=1.,
            ):
        super(MVDeformableFeatureAggregationCuda, self).__init__()
        self.embed_dims = embed_dims
        self.num_groups = num_groups
        self.group_dims = (self.embed_dims // self.num_groups)
        self.num_levels = num_levels
        self.num_cams = num_cams
        self.weights_fc = nn.Linear(self.embed_dims, self.num_groups * self.num_levels * num_pts)
        self.output_proj = nn.Linear(self.embed_dims, self.embed_dims)
        self.learnable_fc = nn.Linear(self.embed_dims, num_pts * 3)
        self.cam_embed = nn.Sequential(
            nn.Linear(26, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.embed_dims),
        )
        self.drop = nn.Dropout(dropout)
        self.im2col_step = im2col_step
        self.bias = bias

    def init_weight(self):
        constant_init(self.weights_fc, val=0.0, bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)
        nn.init.uniform_(self.learnable_fc.bias.data, -self.bias, self.bias)    

    def forward(self, instance_feature, query_pos, feat_flatten, reference_points, spatial_flatten, level_start_index, pc_range, lidar2img_mat, img_metas, cam_params=None, debug_info=None):
        bs, num_anchor = reference_points.shape[:2]
        reference_points = get_ego_pos(reference_points, pc_range)
        key_points = reference_points.unsqueeze(-2) + self.learnable_fc(instance_feature).reshape(bs, num_anchor, -1, 3)

        weights = self._get_weights(instance_feature, query_pos, lidar2img_mat, cam_params)

        features = self.feature_sampling(feat_flatten, spatial_flatten, level_start_index, key_points, weights, lidar2img_mat, img_metas, cam_params=cam_params, debug_info=debug_info)

        output = self.output_proj(features)
        output = self.drop(output) + instance_feature
        return output

    def _get_weights(self, instance_feature, anchor_embed, lidar2img_mat, cam_params=None):
        bs, num_anchor = instance_feature.shape[:2]
        # lidar2img = lidar2img_mat[..., :3, :].flatten(-2)

        rots, trans, intrins, post_rots, post_trans, bda = cam_params
        mln_input = torch.cat([intrins[..., 0, 0:1], intrins[..., 1,1:2], rots.flatten(-2), trans, post_rots.flatten(-2), post_trans], dim=-1)
        

        cam_embed = self.cam_embed(mln_input) # B, N, C
        feat_pos = (instance_feature + anchor_embed).unsqueeze(2)  + cam_embed.unsqueeze(1)
        weights = self.weights_fc(feat_pos).reshape(bs, num_anchor, -1, self.num_groups).softmax(dim=-2)
        weights = weights.reshape(bs, num_anchor, self.num_cams, -1, self.num_groups).permute(0, 2, 1, 4, 3).contiguous()
        return weights.flatten(end_dim=1)

    @force_fp32(apply_to=('feat_flatten', 'key_points'))
    def feature_sampling(self, feat_flatten, spatial_flatten, level_start_index, key_points, weights, lidar2img_mat, img_metas, cam_params=None, debug_info=None):
        bs, num_anchor, _ = key_points.shape[:3]

        rots, trans, intrins, post_rots, post_trans, bda = cam_params
        B, N, _ = trans.shape
        eps = 1e-5
        ogfH, ogfW = img_metas[0]['input_size']
        reference_points = key_points

        # reference_points = debug_info['centers3d'][0][:, :3][None, :, None, :].to(rots.device)
        with autocast(enabled=False):
            reference_points = reference_points[:, None].repeat(1, N, 1, 1, 1)
            reference_points = torch.inverse(bda).view(B, 1, 1, 1, 3,
                          3).matmul(reference_points.unsqueeze(-1)).squeeze(-1)
            reference_points -= trans.view(B, N, 1, 1, 3)
            combine = rots.matmul(torch.inverse(intrins)).inverse()
            points_2d = combine.view(B, N, 1, 1, 3, 3).matmul(reference_points.unsqueeze(-1)).squeeze(-1)
            points_2d = torch.cat([points_2d[..., 0:2] / torch.maximum(
                points_2d[..., 2:3], torch.ones_like(points_2d[..., 2:3])*eps),  points_2d[..., 2:3]], 4
                )
            points_2d = post_rots.view(B, N, 1, 1, 3, 3).matmul(points_2d.unsqueeze(-1)).squeeze(-1)
            points_2d += post_trans.view(B, N, 1, 1, 3) 
            
            # imgs = debug_info['img'][0]
            # import cv2
            # from IPython import embed
            # embed()
            # exit()
            # for i in range(6):
            #     img2 = imgs[i].permute(1, 2, 0).cpu().numpy().astype(np.float32)
            #     img = np.ones([320, 800, 3], dtype=np.float32) * 255
            #     img = img.astype(np.uint8)
            #     for corner in points_2d[0][i]:
            #         corner = corner[0]
            #         if (0<corner[0]<ogfW) & (0<corner[1]<ogfH):
                        
            #             corner = corner.cpu().numpy()[:2].astype(np.int)
            #             print(corner)
            #             img = cv2.circle(img, corner, 2, (61, 102, 255))
            #     img = np.concatenate((img2, img), axis=0)
            #     cv2.imwrite(f'{i}=.png', img[:, :,::-1])
            points_2d[..., 0] /= ogfW
            points_2d[..., 1] /= ogfH

            points_2d = points_2d.flatten(end_dim=1) #[b*6, 900, 13, 2]
            points_2d = points_2d[:, :, None, None, :, :].repeat(1, 1, self.num_groups, self.num_levels, 1, 1)

            bn, num_value, _ = feat_flatten.size()
            feat_flatten = feat_flatten.reshape(bn, num_value, self.num_groups, -1)
            # attention_weights = weights * mask
            output = MultiScaleDeformableAttnFunction.apply(
                    feat_flatten, spatial_flatten, level_start_index, points_2d,
                    weights, self.im2col_step)
        
            output = output.reshape(bs, self.num_cams, num_anchor, -1)

        return output.sum(1)


from mmcv.cnn.bricks.transformer import MultiheadAttention
@ATTENTION.register_module()
class SparseBEVSelfAttention(BaseModule):
    def __init__(self, embed_dims=256, num_heads=8, dropout=0.1, pc_range=[], init_cfg=None, batch_first=True, **kwargs):
        super().__init__(init_cfg)
        self.pc_range = pc_range
        self.embed_dims = embed_dims
        self.attention = MultiheadAttention(embed_dims, num_heads, dropout, batch_first=batch_first)
        self.gen_tau = nn.Linear(embed_dims, num_heads)

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.gen_tau.weight)
        nn.init.uniform_(self.gen_tau.bias, 0.0, 2.0)

    def forward(self,
                    query,
                    temp_key,
                    temp_value,
                    identity,
                    query_pos=None,
                    key_pos=None,
                    attn_mask=None,
                    key_padding_mask=None,
                    reference_points=None,
                    temp_reference_points=None,
                    pc_range=None,
                    **kwargs):
        dist = self.calc_points_dists(reference_points, temp_reference_points, pc_range)
        tau = self.gen_tau(query)  # [B, Q, 8]
        tau = tau.permute(0, 2, 1)  # [B, 8, Q]
        dist_attn_mask = dist[:, None, :, :] * tau[..., None]  # [B, 8, Q, Q]
        if attn_mask is not None:
            dist_attn_mask[:, :, attn_mask] = float('-inf')
        dist_attn_mask = dist_attn_mask.flatten(0, 1)  # [Bx8, Q, Q]

        return self.attention(query,
                    temp_key,
                    temp_value,
                    identity,
                    query_pos,
                    key_pos,
                    dist_attn_mask,)

    @torch.no_grad()
    def calc_points_dists(self, reference_points, temp_reference_points, pc_range):
    
        reference_points = get_ego_pos(reference_points, pc_range)[..., :2] 
        temp_reference_points = get_ego_pos(temp_reference_points, pc_range)[..., :2]
        dist = []
        for b in range(reference_points.shape[0]):
            dist_b = torch.norm(reference_points[b].reshape(-1, 1, 2) - temp_reference_points[b].reshape(1, -1, 2), dim=-1)
            dist.append(dist_b[None, ...])

        dist = torch.cat(dist, dim=0)  # [B, Q, Q]
        dist = -dist

        return dist


from mmcv.cnn.bricks.transformer import MultiheadAttention
@ATTENTION.register_module()
class MotionSelfAttention(BaseModule):
    def __init__(self, embed_dims=256, num_heads=8, dropout=0.1, pc_range=[], init_cfg=None, batch_first=True, dist_func_type='ADE', consider_map_quality=True, **kwargs):
        super().__init__(init_cfg)
        self.pc_range = pc_range
        self.embed_dims = embed_dims
        self.attention = MultiheadAttention(embed_dims, num_heads, dropout, batch_first=batch_first)
        self.gen_tau = nn.Linear(embed_dims, num_heads)
        self.dist_func_type = dist_func_type
        self.consider_map_quality = consider_map_quality
        if self.consider_map_quality and dist_func_type != 'ADE':
            self.map_alpha = nn.Parameter(
                torch.tensor([0.5]), requires_grad=False
            )

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.gen_tau.weight)
        nn.init.uniform_(self.gen_tau.bias, 0.0, 2.0)

    def forward(self,
                    query,
                    key,
                    value,
                    identity,
                    query_pos=None,
                    key_pos=None,
                    attn_mask=None,
                    key_padding_mask=None,
                    reference_points_q=None,
                    reference_points_v=None,
                    pc_range=None,
                    map_scores=None,
                    **kwargs):
        
        func_mapper = {
            'ADE': self.calc_ADE,
            'MDE': self.calc_MDE,
            'MDE_v2': self.calc_MDE_v2
        }
        dist_func = func_mapper[self.dist_func_type]
        dist = dist_func(reference_points_q, reference_points_v, pc_range, map_scores=map_scores)
        tau = self.gen_tau(query)  # [B, Q, 8]
        tau = tau.permute(0, 2, 1)  # [B, 8, Q]
        dist_attn_mask = dist[:, None, :, :] * tau[..., None]  # [B, 8, Q, Q]
        if attn_mask is not None:
            dist_attn_mask[:, :, attn_mask] = float('-inf')
        dist_attn_mask = dist_attn_mask.flatten(0, 1)  # [Bx8, Q, Q]

        return self.attention(query,
                    key,
                    value,
                    identity,
                    query_pos,
                    key_pos,
                    dist_attn_mask,)

    @torch.no_grad()
    def calc_ADE(self, reference_points_q, reference_points_v, pc_range, **kwargs):
        """average distance"""
        dist = []
        code_size = reference_points_q.size(-1)
        n_points = reference_points_q.size(-2)
        for b in range(reference_points_q.shape[0]):
            dist_b = torch.norm(reference_points_q[b].reshape(-1, 1, n_points, code_size) - reference_points_v[b].reshape(1, -1, n_points, code_size), dim=-1)
            dist.append(dist_b[None, ...].mean(-1))
        dist = torch.cat(dist, dim=0)  # [B, Q, K]
        dist = -dist
        return dist


    @torch.no_grad()
    def calc_MDE(self, reference_points_q, reference_points_v, pc_range, map_scores=None):
        """
        mim mean distance between the map lane and traj.
        """
        reference_points_q = reference_points_q[..., :2]
        q_shape = reference_points_q.shape
        v_shape = reference_points_v.shape
        reference_points_q = reference_points_q.flatten(1, 2)
        reference_points_v = reference_points_v.flatten(1, 2)
        
        dist = []
        code_size = reference_points_q.size(-1)
        for b in range(reference_points_q.shape[0]):
            dist_b = torch.norm(reference_points_q[b].reshape(-1, 1, code_size) - reference_points_v[b].reshape(1, -1, code_size), dim=-1)
            dist.append(dist_b[None, ...])
        dist = torch.cat(dist, dim=0)  # [B, Q, K]
        dist = dist.view(q_shape[0], q_shape[1], q_shape[2], v_shape[1], v_shape[2])
        dist = dist.min(-1).values.sum(2)
        
        if self.consider_map_quality and map_scores is not None:
            map_scores = map_scores.sigmoid().max(-1)[0] # smaller, better
            map_scores = torch.round(1-map_scores, decimals=1) + self.map_alpha
            dist = dist * map_scores.unsqueeze(1)
            
        dist = -dist

        return dist

    @torch.no_grad()
    def calc_MDE_v2(self, reference_points_q, reference_points_v, pc_range, map_scores=None):
        """
        mim mean distance between the map lane and traj.
        """
        reference_points_q = reference_points_q[..., :2]
        q_shape = reference_points_q.shape
        v_shape = reference_points_v.shape
        reference_points_q = reference_points_q.flatten(1, 2)
        reference_points_v = reference_points_v.flatten(1, 2)
        
        dist = []
        code_size = reference_points_q.size(-1)
        for b in range(reference_points_q.shape[0]):
            dist_b = torch.norm(reference_points_q[b].reshape(-1, 1, code_size) - reference_points_v[b].reshape(1, -1, code_size), dim=-1)
            dist.append(dist_b[None, ...])
        dist = torch.cat(dist, dim=0)  # [B, Q, K]
        dist = dist.view(q_shape[0], q_shape[1], q_shape[2], v_shape[1], v_shape[2])
        dist = dist.min(-1).values.mean(2)
        dist[dist>5] = 1000
        
        if self.consider_map_quality and map_scores is not None:
            map_scores = map_scores.sigmoid().max(-1)[0] # smaller, better
            # map_scores = torch.round(1-map_scores, decimals=1) + self.map_alpha
            dist[map_scores.unsqueeze(1)<0.2] = 1000
        dist = -dist
        return dist

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class CustomTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR3D transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default: `LN`.
    """

    def __init__(self, *args, return_intermediate=False, **kwargs):
        super(CustomTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False

    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                key_padding_mask=None,
                *args,
                **kwargs):
        """Forward function for `Detr3DTransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        intermediate = []
        for lid, layer in enumerate(self.layers):
            query = layer(
                query=query,
                key=key,
                value=value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_masks=attn_masks,
                key_padding_mask=key_padding_mask,
                *args,
                **kwargs)

            if self.return_intermediate:
                intermediate.append(query)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return query