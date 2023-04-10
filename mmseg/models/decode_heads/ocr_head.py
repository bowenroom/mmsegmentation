# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from ..utils import SelfAttentionBlock as _SelfAttentionBlock
from ..utils import resize
from .cascade_decode_head import BaseCascadeDecodeHead


class SpatialGatherModule(nn.Module):
    """Aggregate the context features according to the initial predicted
    probability distribution.

    Employ the soft-weighted method to aggregate the context.
    """

    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, feats, probs):
        """Forward function."""
        batch_size, num_classes, height, width = probs.size()
        channels = feats.size(1)
        probs = probs.view(batch_size, num_classes, -1)
        feats = feats.view(batch_size, channels, -1)
        # [batch_size, height*width, num_classes]
        feats = feats.permute(0, 2, 1)
        # [batch_size, channels, height*width]
        probs = F.softmax(self.scale * probs, dim=2)
        # [batch_size, channels, num_classes]
        ocr_context = torch.matmul(probs, feats)
        ocr_context = ocr_context.permute(0, 2, 1).contiguous().unsqueeze(3)
        return ocr_context


class ObjectAttentionBlock(_SelfAttentionBlock):
    """Make a OCR used SelfAttentionBlock."""

    def __init__(self, in_channels, channels, scale, conv_cfg, norm_cfg,
                 act_cfg):
        if scale > 1:
            query_downsample = nn.MaxPool2d(kernel_size=scale)
        else:
            query_downsample = None
        super().__init__(
            key_in_channels=in_channels,
            query_in_channels=in_channels,
            channels=channels,
            out_channels=in_channels,
            share_key_query=False,
            query_downsample=query_downsample,
            key_downsample=None,
            key_query_num_convs=2,
            key_query_norm=True,
            value_out_num_convs=1,
            value_out_norm=True,
            matmul_norm=True,
            with_out=True,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.bottleneck = ConvModule(
            in_channels * 2,
            in_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, query_feats, key_feats):
        """Forward function."""
        context = super().forward(query_feats, key_feats)
        output = self.bottleneck(torch.cat([context, query_feats], dim=1))
        if self.query_downsample is not None:
            output = resize(query_feats)

        return output


@MODELS.register_module()
class OCRHead(BaseCascadeDecodeHead):
    """Object-Contextual Representations for Semantic Segmentation.

    This head is the implementation of `OCRNet
    <https://arxiv.org/abs/1909.11065>`_.

    Args:
        ocr_channels (int): The intermediate channels of OCR block.
        scale (int): The scale of probability map in SpatialGatherModule in
            Default: 1.
    """

<<<<<<< HEAD
    def __init__(self, ocr_channels, scale=1, use_sre = False, use_sra = False,  **kwargs):
        super(OCRHead, self).__init__(**kwargs)
=======
    def __init__(self, ocr_channels, scale=1, **kwargs):
        super().__init__(**kwargs)
>>>>>>> upstream/main
        self.ocr_channels = ocr_channels
        self.scale = scale
        self.object_context_block = ObjectAttentionBlock(
            self.channels,
            self.ocr_channels,
            self.scale,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.spatial_gather_module = SpatialGatherModule(self.scale)

        self.bottleneck = ConvModule(
            self.in_channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, inputs, prev_output):
        """Forward function."""
        x = self._transform_inputs(inputs)
        
        feats = self.bottleneck(x)
        # feas are the output from backbone(pixel representations)
        # prev_output are the output from FCNHead(soft object regions)
        context = self.spatial_gather_module(feats, prev_output)
        object_context = self.object_context_block(feats, context)
        output = self.cls_seg(object_context)

        return output

# DASPP in SRE module
#%%
# EACM
class Conv_19(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv_19, self).__init__()
        self.mask = nn.Parameter(torch.Tensor(get_mask_19(in_channels, out_channels)), requires_grad=False)
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, 3, 3), requires_grad=True)
        self.weight.data.mul_(self.mask.data)

    def forward(self, x):
        weight = torch.mul(self.weight, self.mask)
        x = torch.nn.functional.conv2d(x, weight, bias=None, stride=1, padding=1, groups=1)
        return x

def get_mask_19(in_channels, out_channels, kernel_size=3):
    mask = np.zeros((out_channels, in_channels, 3, 3))
    for c in range(kernel_size):
        mask[:, :, c, c] = 1.
    return mask


class Conv_37(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv_37, self).__init__()
        self.mask = nn.Parameter(torch.Tensor(get_mask_37(in_channels, out_channels)), requires_grad=False)
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, 3, 3), requires_grad=True)
        self.weight.data.mul_(self.mask.data)

    def forward(self, x):
        weight = torch.mul(self.weight, self.mask)
        x = torch.nn.functional.conv2d(x, weight, bias=None, stride=1, padding=1, groups=1)
        return x

def get_mask_37(in_channels, out_channels, kernel_size=3):
    mask = np.zeros((out_channels, in_channels, 3, 3))
    for c in range(kernel_size):
        mask[:, :, 2-c, c] = 1.
    return mask
class EACM(nn.Module):
    def __init__(self, in_channels, out_channels, padding_mode='zeros'):
        super(EACM, self).__init__()


        self.square_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=(3, 3), stride=1,
                                     padding=1, bias=False, padding_mode=padding_mode)

        self.ver_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                  kernel_size=(3, 1), stride=1,
                                  padding=(1, 0), bias=False, padding_mode=padding_mode)

        self.hor_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                  kernel_size=(1, 3), stride=1, 
                                  padding=(0, 1), bias=False, padding_mode=padding_mode)

        self.diag19_conv = Conv_19(in_channels=in_channels, out_channels=out_channels)

        self.diag37_conv = Conv_37(in_channels=in_channels, out_channels=out_channels)


    def forward(self, input):
        square_outputs = self.square_conv(input)
        vertical_outputs = self.ver_conv(input)
        horizontal_outputs = self.hor_conv(input)
        diag19_outputs = self.diag19_conv(input)
        diag37_outputs = self.diag37_conv(input)

        return square_outputs + vertical_outputs + horizontal_outputs + diag19_outputs + diag37_outputs