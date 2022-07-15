import math
import warnings
from numpy import append

import torch
import torch.nn as nn
from torch.nn import functional as F
from mmcv.cnn import (Conv2d, Scale, build_activation_layer, build_norm_layer,
                      constant_init, normal_init, trunc_normal_init)
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import MultiheadAttention
from mmcv.runner import BaseModule, ModuleList, Sequential, _load_checkpoint
from ..builder import BACKBONES
from .mit import MixVisionTransformer
from mmcls.models import ConvNeXt


@BACKBONES.register_module()
class CMF(BaseModule):

    def __init__(self,
                 in_channels=4,
                 arch='tiny',
                 out_indices=[0, 1, 2, 3],
                 drop_path_rate=0.4,
                 layer_scale_init_value=1.0,
                 gap_before_final_norm=False,
                 weight=0.5,
                 overlap=True,
                 attention_type='dsa-add',
                 same_branch=True,
                 backbone='ConvNext',
                 **kwargs):
        super(CMF, self).__init__()
        assert (in_channels == 4)

        self.num_heads = kwargs["num_heads"]
        self.num_stages = 4
        # self.overlap = overlap
        # self.weight = weight
        init_cfg = kwargs["init_cfg"]

        if (backbone == "ConvNext"):
            self.color = ConvNeXt(arch=arch, in_channels=in_channels, out_indices=out_indices, drop_path_rate=drop_path_rate,
                                  layer_scale_init_value=layer_scale_init_value, gap_before_final_norm=gap_before_final_norm, init_cfg=init_cfg)
            self.embed_dims = kwargs["embed_dims"]
        else:
            raise NotImplementedError("{} backbone is not supported".format(
                self.backbone))

        if same_branch:
            if (backbone == "ConvNext"):
                self.dsm = ConvNeXt(arch=arch, in_channels=1, out_indices=out_indices, drop_path_rate=drop_path_rate,
                                    layer_scale_init_value=layer_scale_init_value, gap_before_final_norm=gap_before_final_norm, init_cfg=init_cfg)
            else:
                raise NotImplementedError(
                    "{} backbone is not supported".format(self.backbone))
        else:
            self.dsm = DepthDownsample(
                1,
                embed_dims=self.embed_dims,
                num_heads=self.num_heads,
                overlap=self.overlap,
                pretrained=kwargs["pretrained"]
                if "pretrained" in kwargs.keys() else None)

        self.odfm = ModuleList()
        for i in range(self.num_stages):
            embed_dims_i = self.embed_dims * self.num_heads[i]
            if attention_type == 'dsa-concat':

                pass  # self.attention_type == 'none'  just add

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m.weight, std=.02)
                if m.bias is not None:
                    constant_init(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m.bias, 0)
                constant_init(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                normal_init(m.weight, 0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    constant_init(m.bias, 0)
        # load pretrained model if exists
        self.color.init_weights()
        # self.dsm.init_weights()

    def forward(self, x):
        c = x[:, :3]
        d = x[:, 3:]
        c_outs = self.color(c)
        d_outs = self.dsm(d)

        outs = []
        for i in range(self.num_stages):
            c, d = c_outs[i], d_outs[i]
            if (len(self.odfm) != 0):
                out = self.odfm[i](c, d)
                outs.append(out)
            else:
                outs.append(c_outs[i] + d_outs[i])
        return outs
