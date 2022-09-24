# import packages
# %%
import argparse
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
# from pyrsistent import T
# def show_result(imagePath,config_file,checkpoint_file):
#     axs = get_grid(5,1,5,figsize = (50, 10))
#     image = mmcv.imread(imagePath)
#     gtImage = mmcv.imread(imagePath.replace('img_dir','ann_dir'),flag='grayscale', channel_order='rgb')
#     dsmImage = mmcv.imread(imagePath.replace('img_dir','dsm_dir').replace('.png','.tiff'),flag=-1)
#     model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
#     result = inference_segmentor(model, imagePath)
#     result2 = inference_segmentor(model, fogPath)
#     # fogImage = mmcv.imread(fogPath)
#     inferImage =  model.show_result(
#     imagePath, result, palette=get_palette('potsdam'), show=False, opacity=1)
#     inferImage2 =  model.show_result(
#         fogPath, result2, palette=get_palette('potsdam'), show=False, opacity=1)
#     [i.set_axis_off() for i in axs]
#     axs[0].imshow(image)
#     axs[2].imshow(gtImage,cmap = my_cmap)
#     axs[1].imshow(dsmImage,cmap = 'gray')
#     axs[3].imshow(mmcv.bgr2rgb(inferImage))
#     axs[4].imshow(mmcv.bgr2rgb(inferImage2))
# show_result(imagePath,config_file,checkpoint_file)
# %%
ann_path = '../tempDataset/potsdam/ann_dir/val/2_13_4608_0_5120_512.png'
img_path = '../tempDataset/potsdam/dsm_dir/val/2_13_4608_0_5120_512.tiff'


def colormap():
    #  #FFFFFF #0000FF #00FFFF #00FF00 #FFFF00 #FF0000
    # cdict = ['#FFFFFF', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00']
    #
    cdict = ['#000000', '#FFFFFF', '#FF0000',
             '#FFFF00',  '#00FF00', '#00FFFF', '#0000FF']
    # 按照上面定义的colordict，将数据分成对应的部分，indexed：代表顺序
    return colors.ListedColormap(cdict, 'from_list')


color_map = np.array([[0, 0, 0], [255, 255, 255], [255, 0, 0],
                      [255, 255, 0], [0, 255, 0], [0, 255, 255],
                      [0, 0, 255]])

# define my own pixel color paletter in the matplotlib
my_cmap = colormap()

# %%
def show_random_result(ann_path):
    axs = get_grid(2,1,2,figsize = (20, 10))
    ann_path = get_image_files(ann_path)[random.randint(0, 199)]
    ann_img = mmcv.imread(ann_path, flag='grayscale')
    ori_img = mmcv.imread(str(ann_path).replace('ann_dir','img_dir'),channel_order='rgb')
    print(f'unique color encoding of this image is {np.unique(ann_img)}')
    [i.set_axis_off() for i in axs]
    axs[0].imshow(ori_img)
    axs[1].imshow(ann_img,cmap = my_cmap)
show_random_result('/home/swp/paperCode/IGRLCode/mmf/tempDataset/potsdam/ann_dir')


