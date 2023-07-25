# Copyright (c) OpenMMLab. All rights reserved.
import random
import time
from collections import defaultdict
from functools import partial
from itertools import chain
from typing import Sequence

import dgl
import dgl.backend.pytorch.sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import (EdgeConv, GATConv, GatedGraphConv, GATv2Conv,
                            GCN2Conv, GINConv, GMMConv, GraphConv, SAGEConv)
from dgl.nn.pytorch.factory import KNNGraph, RadiusGraph
from mmcv.cnn.bricks import (NORM_LAYERS, DropPath, build_activation_layer,
                             build_norm_layer)
from mmcv.runner import BaseModule
from mmcv.runner.base_module import ModuleList, Sequential

from ..builder import BACKBONES
from .base_backbone import BaseBackbone


@NORM_LAYERS.register_module('LN2d_Graph')
class LayerNorm2d(nn.LayerNorm):
    """LayerNorm on channels for 2d images.

    Args:
        num_channels (int): The number of channels of the input tensor.
        eps (float): a value added to the denominator for numerical stability.
            Defaults to 1e-5.
        elementwise_affine (bool): a boolean value that when set to ``True``,
            this module has learnable per-element affine parameters initialized
            to ones (for weights) and zeros (for biases). Defaults to True.
    """

    def __init__(self, num_channels: int, **kwargs) -> None:
        super().__init__(num_channels, **kwargs)
        self.num_channels = self.normalized_shape[0]

    def forward(self, x):
        assert x.dim() == 4, 'LayerNorm2d only supports inputs with shape ' \
            f'(N, C, H, W), but got tensor with shape {x.shape}'
        return F.layer_norm(
            x.permute(0, 2, 3, 1), self.normalized_shape, self.weight,
            self.bias, self.eps).permute(0, 3, 1, 2)


class ConvNeXtBlock(BaseModule):
    """ConvNeXt Block.

    Args:
        in_channels (int): The number of input channels.
        norm_cfg (dict): The config dict for norm layers.
            Defaults to ``dict(type='LN2d_Graph', eps=1e-6)``.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        mlp_ratio (float): The expansion ratio in both pointwise convolution.
            Defaults to 4.
        linear_pw_conv (bool): Whether to use linear layer to do pointwise
            convolution. More details can be found in the note.
            Defaults to True.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        layer_scale_init_value (float): Init value for Layer Scale.
            Defaults to 1e-6.

    Note:
        There are two equivalent implementations:

        1. DwConv -> LayerNorm -> 1x1 Conv -> GELU -> 1x1 Conv;
           all outputs are in (N, C, H, W).
        2. DwConv -> LayerNorm -> Permute to (N, H, W, C) -> Linear -> GELU
           -> Linear; Permute back

        As default, we use the second to align with the official repository.
        And it may be slightly faster.
    """

    def __init__(self,
                 in_channels,
                 norm_cfg=dict(type='LN2d_Graph', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 mlp_ratio=4.,
                 linear_pw_conv=True,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=7,
            padding=3,
            groups=in_channels)

        self.linear_pw_conv = linear_pw_conv
        self.norm = build_norm_layer(norm_cfg, in_channels)[1]

        mid_channels = int(mlp_ratio * in_channels)
        if self.linear_pw_conv:
            # Use linear layer to do pointwise conv.
            pw_conv = nn.Linear
        else:
            pw_conv = partial(nn.Conv2d, kernel_size=1)

        self.pointwise_conv1 = pw_conv(in_channels, mid_channels)
        self.act = build_activation_layer(act_cfg)
        self.pointwise_conv2 = pw_conv(mid_channels, in_channels)

        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((in_channels)),
            requires_grad=True) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.depthwise_conv(x)
        x = self.norm(x)

        if self.linear_pw_conv:
            x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)

        x = self.pointwise_conv1(x)
        x = self.act(x)
        x = self.pointwise_conv2(x)

        if self.linear_pw_conv:
            x = x.permute(0, 3, 1, 2)  # permute back

        if self.gamma is not None:
            x = x.mul(self.gamma.view(1, -1, 1, 1))

        x = shortcut + self.drop_path(x)
        return x


class ContextGraphBlock(BaseModule):
    """ContextGraph Block.

    Args:
        in_channels
        patch_size
        graph_heads=4
        graph_residual=True
        radius=3
        p=2
        drop_edge_ratio=0.2
        act_cfg=dict(type='GELU')
        norm_cfg=dict(type='LN2d_Graph', eps=1e-6)
    Note:
        
    """

    def __init__(self,
                 in_channels,
                 patch_size,
                 graph_heads=4,
                 graph_residual=True,
                 radius=2,    # 1.1 or 1.5 or 2
                 p=2,
                 k=12,
                 drop_edge_ratio=0.4,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='BN2d', eps=1e-6)):
        super().__init__()
        self.patch_size = patch_size
        self.drop_edge_ratio = drop_edge_ratio
        # print("drop_edge_ratio: {}".format(drop_edge_ratio))
        self.act = build_activation_layer(act_cfg)
        self.norm = build_norm_layer(norm_cfg, in_channels)[1]

        self.attention = True
        
        if self.attention:
            # GAT
            self.graph_layer = GATConv(
                in_channels, 
                in_channels, 
                graph_heads, 
                residual=graph_residual
            )
            # GATv2
            # self.graph_layer = GATv2Conv(
            #     in_channels, 
            #     in_channels, 
            #     graph_heads, 
            #     residual=graph_residual
            # )
            self.attention_head_fc = nn.Linear(
                in_features=in_channels * graph_heads, 
                out_features=in_channels * (patch_size ** 2)
            )
            self.batch_norm = nn.BatchNorm1d(num_features=in_channels * graph_heads)
            # self.layer_norm = nn.LayerNorm(in_channels * graph_heads)
        else:
            # GCN
            # self.graph_layer = GraphConv(
            #     in_channels * (patch_size ** 2), 
            #     in_channels * (patch_size ** 2),
            #     norm='both',
            #     weight=True,
            #     bias=True
            # )

            # GIN
            # lin = torch.nn.Linear(
            #     in_channels * (patch_size ** 2),
            #     in_channels * (patch_size ** 2)
            # )
            # self.graph_layer = GINConv(
            #     apply_func=lin,
            #     aggregator_type='sum',
            #     activation=F.gelu
            # )

            # GraphSAGE
            self.graph_in_Linear = nn.Linear(in_channels * (patch_size ** 2), in_channels)
            self.graph_in_norm = nn.BatchNorm1d(num_features=in_channels)
            # self.graph_layer = SAGEConv(
            #     in_channels,
            #     in_channels,
            #     aggregator_type='mean',
            #     feat_drop=0.5
            # )
            self.graph_out_norm = nn.BatchNorm1d(num_features=in_channels)
            self.graph_out_Linear = nn.Linear(in_channels, in_channels * (patch_size ** 2))

            # EdgeConv
            self.graph_layer = EdgeConv(
                in_channels,
                in_channels,
                batch_norm=False
            )
        
        # self.radius_graph = RadiusGraph(radius, p=p, self_loop=True)
        # self.radius_graph = RadiusGraph(radius, p=p)
        self.radius_graph = KNNGraph(k=k)
        
        
    def forward(self, x: torch.Tensor):
        shortcut = x
        N, C, H, W = x.shape
        pad_H, pad_W = 0, 0
        if H % self.patch_size != 0:
            pad_H = self.patch_size - H % self.patch_size
        if W % self.patch_size != 0:
            pad_W = self.patch_size - W % self.patch_size
        
        x = F.pad(x, pad=(0, pad_W, 0, pad_H), mode='constant', value=0)
        new_H = pad_H + H
        new_W = pad_W + W
        
        device = x.device
        patch_num_H, patch_num_W = new_H // self.patch_size, new_W // self.patch_size

        #-------------------------------GetPatch-------------------------------#
        x = x.view(N, C, patch_num_H, self.patch_size, patch_num_W, self.patch_size)
        x = x.permute(0, 2, 4, 3, 5, 1).clone().contiguous()
        
        if self.attention:
            x = x.view(N * patch_num_H * patch_num_W, self.patch_size ** 2, -1)
            x = x.sum(dim=1)
        else:
            x = x.view(N * patch_num_H * patch_num_W, -1)

        #-------------------------------GetBatchEdge-------------------------------#
        node_idx = torch.tensor([[h, w] for h in range(patch_num_H) for w in range(patch_num_W)], dtype=torch.float)
        rg = self.radius_graph(node_idx)
        edges = rg.edges()
        num_nodes = rg.num_nodes()
        total_edges_u = torch.cat([edges[0] + num_nodes * i for i in range(N)])
        total_edges_v = torch.cat([edges[1] + num_nodes * i for i in range(N)])

        #-------------------------------GetAllEdge-------------------------------#
        batch_radius_graph = dgl.graph((total_edges_u, total_edges_v))
        batch_radius_graph = batch_radius_graph.to(device)
        all_num_edges = batch_radius_graph.num_edges()

        #-------------------------------Remove-------------------------------#
        remove_edge = []
        num_drop = int(self.drop_edge_ratio * all_num_edges)
        for j in range(num_drop):
            remove_edge.append(int((all_num_edges - 1) * random.random()))
        batch_radius_graph.remove_edges(remove_edge)
        
        batch_radius_graph = dgl.add_self_loop(batch_radius_graph)
        # batch_radius_graph.add_self_loop()

        #-------------------------------MessageAggregation-------------------------------#
        self.graph_layer = self.graph_layer.to(device)
        if self.attention:
            x = self.graph_layer(batch_radius_graph, x)
            x = x.view(x.shape[0], -1)
            x = self.batch_norm(x)
            # x = self.layer_norm(x)
            x = F.leaky_relu(x)

            #-------------------------------CombineHeads-------------------------------#
            x = self.attention_head_fc(x)
        else:
            x = self.graph_in_Linear(x)
            x = self.graph_in_norm(x)
            x = F.leaky_relu(x)
            x = self.graph_layer(batch_radius_graph, x)
            x = self.graph_out_norm(x)
            x = F.leaky_relu(x)
            x = self.graph_out_Linear(x)

        #-------------------------------Reshape:(NxPHxPW, PSxPSxC) -> (N, C, H, W)-------------------------------#
        x = x.view(N, patch_num_H, patch_num_W, self.patch_size, self.patch_size, C)
        x = x.permute(0, 5, 1, 3, 2, 4).clone().contiguous()
        x = x.view(N, C, new_H, new_W)
        x = x[:, :, 0: H, 0: W]
        x = self.norm(x)
        x = self.act(x)

        x = shortcut + x
        return x


@BACKBONES.register_module()
class ConvNeXt_Graph(BaseBackbone):
    """ConvNeXt.

    A PyTorch implementation of : `A ConvNet for the 2020s
    <https://arxiv.org/pdf/2201.03545.pdf>`_

    Modified from the `official repo
    <https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py>`_
    and `timm
    <https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/convnext.py>`_.

    Args:
        arch (str | dict): The model's architecture. If string, it should be
            one of architecture in ``ConvNeXt.arch_settings``. And if dict, it
            should include the following two keys:

            - depths (list[int]): Number of blocks at each stage.
            - channels (list[int]): The number of channels at each stage.

            Defaults to 'tiny'.
        in_channels (int): Number of input image channels. Defaults to 3.
        stem_patch_size (int): The size of one patch in the stem layer.
            Defaults to 4.
        norm_cfg (dict): The config dict for norm layers.
            Defaults to ``dict(type='LN2d_Graph', eps=1e-6)``.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        linear_pw_conv (bool): Whether to use linear layer to do pointwise
            convolution. Defaults to True.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        layer_scale_init_value (float): Init value for Layer Scale.
            Defaults to 1e-6.
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        frozen_stages (int): Stages to be frozen (all param fixed).
            Defaults to 0, which means not freezing any parameters.
        gap_before_final_norm (bool): Whether to globally average the feature
            map before the final norm layer. In the official repo, it's only
            used in classification task. Defaults to True.
        init_cfg (dict, optional): Initialization config dict
    """  # noqa: E501
    arch_settings = {
        'tiny': {
            'depths': [3, 3, 9, 3],
            'channels': [96, 192, 384, 768],
            'patch_sizes': [8, 4, 2, 1]
        },
        'small': {
            'depths': [3, 3, 27, 3],
            'channels': [96, 192, 384, 768],
            'patch_sizes': [8, 4, 2, 1]
        },
        'base': {
            'depths': [3, 3, 27, 3],
            'channels': [128, 256, 512, 1024],
            'patch_sizes': [8, 4, 2, 1]
        },
        'large': {
            'depths': [3, 3, 27, 3],
            'channels': [192, 384, 768, 1536],
            'patch_sizes': [8, 4, 2, 1]
        },
        'xlarge': {
            'depths': [3, 3, 27, 3],
            'channels': [256, 512, 1024, 2048],
            'patch_sizes': [8, 4, 2, 1]
        },
    }

    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 stem_patch_size=4,
                 norm_cfg=dict(type='LN2d_Graph', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 linear_pw_conv=True,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 out_indices=-1,
                 frozen_stages=0,
                 gap_before_final_norm=True,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'depths' in arch and 'channels' in arch, \
                f'The arch dict must have "depths" and "channels", ' \
                f'but got {list(arch.keys())}.'

        self.depths = arch['depths']
        self.channels = arch['channels']
        self.patch_sizes = arch['patch_sizes']
        assert (isinstance(self.depths, Sequence)
                and isinstance(self.channels, Sequence)
                and len(self.depths) == len(self.channels)), \
            f'The "depths" ({self.depths}) and "channels" ({self.channels}) ' \
            'should be both sequence with the same length.'

        self.num_stages = len(self.depths)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices

        self.frozen_stages = frozen_stages
        self.gap_before_final_norm = gap_before_final_norm

        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        block_idx = 0

        # 4 downsample layers between stages, including the stem layer.
        self.downsample_layers = ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.channels[0],
                kernel_size=stem_patch_size,
                stride=stem_patch_size),
            build_norm_layer(norm_cfg, self.channels[0])[1],
        )
        self.downsample_layers.append(stem)

        # 4 feature resolution stages, each consisting of multiple residual
        # blocks
        self.stages = nn.ModuleList()
        self.graph_stages = nn.ModuleList()

        graph_depths = self.depths
        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]
            patch_size = self.patch_sizes[i]

            if i >= 1:
                downsample_layer = nn.Sequential(
                    LayerNorm2d(self.channels[i - 1]),
                    nn.Conv2d(
                        self.channels[i - 1],
                        channels,
                        kernel_size=2,
                        stride=2),
                )
                self.downsample_layers.append(downsample_layer)

            stage = Sequential(*[
                ConvNeXtBlock(
                    in_channels=channels,
                    drop_path_rate=dpr[block_idx + j],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    linear_pw_conv=linear_pw_conv,
                    layer_scale_init_value=layer_scale_init_value)
                for j in range(depth)
            ])
            graph_stage = Sequential(*[
                ContextGraphBlock(
                    in_channels=channels,
                    patch_size=patch_size
                ) for j in range(graph_depths[i])
            ])
            block_idx += depth

            self.stages.append(stage)
            self.graph_stages.append(graph_stage)

            if i in self.out_indices:
                norm_layer = build_norm_layer(norm_cfg, channels)[1]
                self.add_module(f'norm{i}', norm_layer)

        self._freeze_stages()

    def forward(self, x):
        outs = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            x = stage(x)
            x = self.graph_stages[i](x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                if self.gap_before_final_norm:
                    gap = x.mean([-2, -1], keepdim=True)
                    outs.append(norm_layer(gap).flatten(1))
                else:
                    # The output of LayerNorm2d may be discontiguous, which
                    # may cause some problem in the downstream tasks
                    outs.append(norm_layer(x).contiguous())

        return tuple(outs)

    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            downsample_layer = self.downsample_layers[i]
            stage = self.stages[i]
            downsample_layer.eval()
            stage.eval()
            for param in chain(downsample_layer.parameters(),
                               stage.parameters()):
                param.requires_grad = False

    def train(self, mode=True):
        super(ConvNeXt_Graph, self).train(mode)
        self._freeze_stages()
