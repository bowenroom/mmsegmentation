# test for different modules
# %%
# import packages
from turtle import forward
from mmseg.models.decode_heads.sep_aspp_head import DepthwiseSeparableASPPModule
from mmseg.models.decode_heads import DAHead
from mmseg.models.decode_heads import ASPPHead, DepthwiseSeparableASPPHead
import argparse
from fileinput import filename
import glob
import imp
import math
import os
import os.path as osp
import tempfile
import zipfile
import matplotlib.colors as colors
import mmcv
import numpy as np
# import  fastai related modules
from fastai.basics import *
from fastai.vision.all import *
from timm.models.layers import create_attn

torch.cuda.set_device(1)
# %%
# ecaAttn = create_attn('eca', 3, use_mlp=True)
# input = torch.randn(1, 3, 512, 512)

# out = ecaAttn(input)
# %%

# %%


def to_cuda(module, data):
    module = module.cuda()
    if isinstance(data, list):
        for i in range(len(data)):
            data[i] = data[i].cuda()
    return module, data

# %%


def test_da_head():

    inputs = [torch.randn(1, 16, 23, 23)]
    head = DAHead(in_channels=16, channels=8, num_classes=19, pam_channels=8)
    if torch.cuda.is_available():
        head, inputs = to_cuda(head, inputs)
    outputs = head(inputs)
    assert isinstance(outputs, tuple) and len(outputs) == 3
    for output in outputs:
        assert output.shape == (1, head.num_classes, 23, 23)
    test_output = head.forward_test(inputs, None, None)
    assert test_output.shape == (1, head.num_classes, 23, 23)

# test_da_head()


# %%
channels = 1888
t = int(abs(math.log(channels, 2) + 1 / (1 + math.exp(-channels)) + 1) / 2)
print(t)

# %%
# 显示img,gt,imgFog5
imgName = '/home/swp/paperCode/IGRLCode/mmf/tempDataset/vaihingen/img_dsm_dir/val/area2_1916_0_2428_512.png'
gtName = '/home/swp/paperCode/IGRLCode/mmf/tempDataset/vaihingen/ann_dir/testFog/area2_1916_0_2428_512_fog_1.png'
imgFog = '/home/swp/paperCode/IGRLCode/mmf/tempDataset/vaihingen/img_dsm_dir/testFog/area2_1916_0_2428_512.png'


def showImages():
    fig, axs = get_grid(5, 1, 5, figsize=(50, 10), return_fig=True)
    for i in range(5):
        tempPath = f'/home/swp/paperCode/IGRLCode/mmf/tempDataset/vaihingen/img_dsm_dir/testFog/area2_1916_0_2428_512_fog_{i+1}.png'
        tempImage = mmcv.imread(tempPath, channel_order='rgb')
        [i.set_axis_off() for i in axs]
        axs[i].imshow(tempImage[:, :, :3])
    fig.tight_layout()
    plt.savefig(
        '/home/swp/paperCode/IGRLCode/mmf/swpTest/fogLevels.png', dpi=400)


def colormap():
    #  #FFFFFF #0000FF #00FFFF #00FF00 #FFFF00 #FF0000
    # cdict = ['#FFFFFF', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00']
    #
    cdict = ['#000000', '#FFFFFF', '#FF0000',
             '#FFFF00',  '#00FF00', '#00FFFF', '#0000FF']
    # 按照上面定义的colordict，将数据分成对应的部分，indexed：代表顺序
    return colors.ListedColormap(cdict, 'from_list')


# define my own pixel color paletter in the matplotlib
my_cmap = colormap()


def show_img_gt():
    fig, axs = get_grid(2, 1, 2, figsize=(50, 10), return_fig=True)
    for i in range(2):
        [i.set_axis_off() for i in axs]
    img = mmcv.imread(imgName, channel_order='rgb')
    gt = mmcv.imread(gtName, flag='grayscale')
    axs[0].imshow(img[:, :, :3])
    axs[1].imshow(gt, cmap=my_cmap)
    fig.tight_layout()
    plt.savefig('/home/swp/paperCode/IGRLCode/mmf/swpTest/imgGT.png', dpi=400)


# showImages()
# show_img_gt()
# %%
# design a cotlayer from 'Contextual Transformer Networks for Visual Recognition'
# read a multimodal image
imagPath = '/home/swp/paperCode/IGRLCode/mmf/tempDataset/vaihingen/img_dsm_dir/train/area1_0_0_512_512.png'
img = mmcv.imread(imagPath, flag='unchanged')
img
# %%
optImage = img[:, :, :3]
dsmImage = img[:, :, 3]
input = [torch.randn(1, 3, 256, 256),
         #  torch.randn(1, 3, 512, 256),
         #  torch.randn(1, 3, 256, 512)
         ]
input = torch.randn(1, 3, 256, 256)

# model = DepthwiseSeparableASPPHead(
#     0, 0, in_channels=3, channels=5, num_classes=6, dilations=(1, 12, 24))
model = DepthwiseSeparableASPPModule(
    in_channels=3, channels=5,  dilations=(1, 12, 24),
    act_cfg=dict(type='ReLU'),
    conv_cfg=None,
    norm_cfg=None,)
out1 = model(input)
print(out1[0].shape)
# %%
from mmcls.models.backbones.van import VANBlock
vanModel = VANBlock(5)
out2 = vanModel(out1[0])
print(out2.shape)
# %%
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

pool = GeM()
# %%
out3 = pool(out2)
print(out3.shape)

# %%
from mmseg.models.decode_heads.da_head import PAM
from mmseg.models.decode_heads.da_head import CAM
# pam = PAM(5,3)
# cam = CAM()
# temp = torch.randn(1,5,256,256)
# rgbf1 = pam(temp)
# rgbf2 = cam(temp)

# %%
# cotlayer implementation
from mmcv.cnn import Scale
class Cotlayer(nn.Module):
    def __init__(self,dim,kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.key_embed = nn.Sequential(
            # 通过K*K的卷积提取邻近上下文信息，视作输入X的静态上下文表达
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.value_embed = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size = 1, stride=1, bias=False),  # 1*1的卷积进行Value的编码
            nn.BatchNorm2d(dim)
        )

        factor = 4
        self.attention_embed = nn.Sequential(  # 通过连续两个1*1的卷积计算注意力矩阵
            nn.Conv2d(2 * dim, 2 * dim // factor, 1, bias=False),  # 输入concat后的特征矩阵 Channel = 2*C
            nn.BatchNorm2d(2 * dim // factor),
            nn.ReLU(),
            nn.Conv2d(2 * dim // factor, kernel_size * kernel_size * dim, 1, stride=1)  # out: H * W * (K*K*C)
        )
        self.gamma = Scale(0)
    def forward(self,x):
        bs, c, h, w = x.shape
        k1 = self.key_embed(x)  # shape：bs,c,h,w  提取静态上下文信息得到key
        v = self.value_embed(x).view(bs, c, -1)  # shape：bs,c,h*w  得到value编码
        y = torch.cat([k1, x], dim=1)  # shape：bs,2c,h,w  Key与Query在channel维度上进行拼接进行拼接
        att = self.attention_embed(y)  # shape：bs,c*k*k,h,w  计算注意力矩阵
        att = att.reshape(bs, c, self.kernel_size * self.kernel_size, h, w)
        att = att.mean(2, keepdim=False).view(bs, c, -1)  # shape:bs,c,h*w  求平均降低维度
        k2 = F.softmax(att, dim=-1) * v  # 对每一个Channel进行softmax后
        k2 = k2.view(bs, c, h, w)
        return k1 + self.gamma(k2)
#%%
input = torch.randn(1,3,512,256)
print(input.shape)
model = Cotlayer(3,3)
output = model(input)
print(output.shape)
# %%
