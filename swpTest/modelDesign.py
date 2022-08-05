# test for different modules
# %%
# import packages
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
ecaAttn = create_attn('eca',3,use_mlp=True)
input = torch.randn(1,3,512,512)

out = ecaAttn(input)
# %%

# %%
def to_cuda(module, data):
    module = module.cuda()
    if isinstance(data, list):
        for i in range(len(data)):
            data[i] = data[i].cuda()
    return module, data

# %%

from mmseg.models.decode_heads import DAHead
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

test_da_head()

# %%
channels = 1888
t = int(abs(math.log(channels, 2)+ 1 / (1 + math.exp(-channels))+ 1) / 2)
print(t)

# %%
# 显示img,gt,imgFog5
imgName = '/home/swp/paperCode/IGRLCode/mmf/tempDataset/vaihingen/img_dsm_dir/val/area2_1916_0_2428_512.png'
gtName = '/home/swp/paperCode/IGRLCode/mmf/tempDataset/vaihingen/ann_dir/testFog/area2_1916_0_2428_512_fog_1.png'
imgFog = '/home/swp/paperCode/IGRLCode/mmf/tempDataset/vaihingen/img_dsm_dir/testFog/area2_1916_0_2428_512.png'
def showImages():
    fig,axs = get_grid(5,1,5,figsize = (50, 10),return_fig=True)
    for i in range(5):
        tempPath = f'/home/swp/paperCode/IGRLCode/mmf/tempDataset/vaihingen/img_dsm_dir/testFog/area2_1916_0_2428_512_fog_{i+1}.png'
        tempImage = mmcv.imread(tempPath,channel_order='rgb')
        [i.set_axis_off() for i in axs]
        axs[i].imshow(tempImage[:,:,:3])
    fig.tight_layout()
    plt.savefig('/home/swp/paperCode/IGRLCode/mmf/swpTest/fogLevels.png',dpi=400)

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
    fig,axs = get_grid(2,1,2,figsize = (50, 10),return_fig=True)
    for i in range(2):
        [i.set_axis_off() for i in axs]
    img = mmcv.imread(imgName,channel_order='rgb')
    gt = mmcv.imread(gtName,flag='grayscale')
    axs[0].imshow(img[:,:,:3])
    axs[1].imshow(gt,cmap=my_cmap)
    fig.tight_layout()
    plt.savefig('/home/swp/paperCode/IGRLCode/mmf/swpTest/imgGT.png',dpi=400)

showImages()
show_img_gt()
# %%
