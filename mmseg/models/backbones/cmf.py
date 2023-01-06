from mmcv.cnn import Scale
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
# from mmcv.cnn import (Conv2d, Scale, build_activation_layer, build_norm_layer,
#                       constant_init, normal_init, trunc_normal_init)

from mmcv.runner import BaseModule, ModuleList, Sequential, _load_checkpoint

from ..builder import BACKBONES
from mmcls.models import ConvNeXt
from mmseg.models.decode_heads.sep_aspp_head import DepthwiseSeparableASPPModule
from mmcls.models.backbones.van import VANBlock


@BACKBONES.register_module()
class CMF(BaseModule):

    def __init__(self,
                 in_channels=3,
                 arch='base',
                 out_indices=[0, 1, 2, 3],
                 drop_path_rate=0.4,
                 layer_scale_init_value=1.0,
                 gap_before_final_norm=False,
                 weight=0.5,
                 overlap=True,
                 attention_type='dsa-add',
                 same_branch=True,
                 backbone='ConvNext',
                 input_embeds = [96, 192, 384, 768],
                 frozen_stages=0,
                 **kwargs):
        super(CMF, self).__init__()
        # assert (in_channels == 4)

        self.num_heads = kwargs["num_heads"]
        self.num_stages = 4
        self.frozen_stages = frozen_stages
        # self.overlap = overlap
        # self.weight = weight
        init_cfg = kwargs["init_cfg"]
        # enhance the representation

        self.crems = nn.ModuleList(
           [ CREM(input_embeds[0], 100),
            CREM(input_embeds[1], 100),
            CREM(input_embeds[2], 100),
            CREM(input_embeds[3], 100)]
        )
        self.cmfs = nn.ModuleList([
            Cotlayer(input_embeds[0],3),
            Cotlayer(input_embeds[1],3),
            Cotlayer(input_embeds[2],3),
            Cotlayer(input_embeds[3],3)
        ])

        if (backbone == "ConvNext"):
            self.color = ConvNeXt(arch=arch, in_channels=in_channels, out_indices=out_indices, drop_path_rate=drop_path_rate,
                                  layer_scale_init_value=layer_scale_init_value,frozen_stages=frozen_stages, gap_before_final_norm=gap_before_final_norm, init_cfg=init_cfg)
            self.embed_dims = kwargs["embed_dims"]
        else:
            raise NotImplementedError("{} backbone is not supported".format(
                self.backbone))

    
        if same_branch:
            if (backbone == "ConvNext"):
                self.dsm = ConvNeXt(arch=arch, in_channels=3, out_indices=out_indices, drop_path_rate=drop_path_rate,
                                    layer_scale_init_value=layer_scale_init_value,frozen_stages=frozen_stages, gap_before_final_norm=gap_before_final_norm, init_cfg=init_cfg)
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

    def forward(self, x):
        # detach RGB and DSM data
        c = x[:, :3]
        d = x[:, 3:]
        d = torch.cat((d, d, d), dim=1)

        c_outs = self.color(c)
        d_outs = self.dsm(d)

        outs = []
        for i in range(self.num_stages):
            c = c_outs[i]
            d = d_outs[i]

            # enhance the representation
            c2, d2 =self.crems[i](c,d)

            if (len(self.odfm) != 0):
                out = self.odfm[i](c, d)
                outs.append(out)
            else:
                # outs.append(c_outs[i] + d_outs[i])
                # multimodal fusion
                outs.append(self.cmfs[i](c2,d2))
                # outs.append(c)
        return outs
# %%

class CREM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        # self.in_channel = in_channel
        # local branch
        self.daspp = DepthwiseSeparableASPPModule(
            in_channels=in_channel, channels=out_channel,  dilations=(
                1, 12, 24),
            act_cfg=dict(type='ReLU'),
            conv_cfg=None,
            norm_cfg=None)

        self.van = VANBlock(out_channel*3)
        self.bottleneck = nn.Conv2d(
            in_channels=out_channel*3,
            out_channels=in_channel,
            kernel_size=1
        )
        self.gempool = GeM()

    def forward(self, x, d):
        rgb_aspp_out = []
        dsm_aspp_out = []

        #####RGB######
        # local branch
        rgb_aspp_out.extend(self.daspp(x))
        rgb_aspp_out = torch.cat(rgb_aspp_out, dim=1)

        rgb_van_out = self.van(rgb_aspp_out)
        rgb_van_out = self.bottleneck(rgb_van_out)
        # global branch
        rgb_pool_out = self.gempool(x)
        rgb_crem_out = rgb_pool_out * rgb_van_out

        #####DSM######
        # local branch
        dsm_aspp_out.extend(self.daspp(d))
        dsm_aspp_out = torch.cat(dsm_aspp_out, dim=1)

        dsm_van_out = self.van(dsm_aspp_out)
        dsm_van_out = self.bottleneck(dsm_van_out)
        # global branch
        dsm_pool_out = self.gempool(d)
        dsm_crem_out = dsm_pool_out * dsm_van_out

        return rgb_crem_out, dsm_crem_out


# %%
# local branch which is composed of DASPP and VAN
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

# %%


class Cotlayer(nn.Module):
    def __init__(self, dim, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.key_embed = nn.Sequential(
            # 通过K*K的卷积提取邻近上下文信息，视作输入X的静态上下文表达
            nn.Conv2d(dim, dim, kernel_size=kernel_size,
                      padding=1, stride=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.value_embed = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1,
                      bias=False),  # 1*1的卷积进行Value的编码
            nn.BatchNorm2d(dim)
        )

        factor = 4
        self.attention_embed = nn.Sequential(  # 通过连续两个1*1的卷积计算注意力矩阵
            # 输入concat后的特征矩阵 Channel = 2*C
            nn.Conv2d(2 * dim, 2 * dim // factor, 1, bias=False),
            nn.BatchNorm2d(2 * dim // factor),
            nn.ReLU(),
            nn.Conv2d(2 * dim // factor, kernel_size * kernel_size * \
                      dim, 1, stride=1)  # out: H * W * (K*K*C)
        )
        self.gamma = Scale(0)

    def forward(self, x, d):
        bs, c, h, w = x.shape
        k1 = self.key_embed(x)  # shape：bs,c,h,w  提取静态上下文信息得到key
        # v = self.value_embed(x).view(bs, c, -1)  # shape：bs,c,h*w  得到value编码
        # shape：bs,c,h*w  得到value编码(提取dsm中的信息)
        v = self.value_embed(d).view(bs, c, -1)
        # y = torch.cat([k1, x], dim=1)  # shape：bs,2c,h,w  Key与Query在channel维度上进行拼接进行拼接
        # shape：bs,2c,h,w  Key与dsm Query在channel维度上进行拼接进行拼接
        y = torch.cat([k1, d], dim=1)
        att = self.attention_embed(y)  # shape：bs,c*k*k,h,w  计算注意力矩阵
        att = att.reshape(bs, c, self.kernel_size * self.kernel_size, h, w)
        att = att.mean(2, keepdim=False).view(
            bs, c, -1)  # shape:bs,c,h*w  求平均降低维度
        k2 = F.softmax(att, dim=-1) * v  # 对每一个Channel进行softmax后
        k2 = k2.view(bs, c, h, w)
        return k1 + self.gamma(k2)
