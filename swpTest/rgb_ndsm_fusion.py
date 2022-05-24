# %%
# import packages
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
# %%
tempRootPath = Path('tempDataTest')
vaihingenRGBPath = tempRootPath/'vaihingen/img_dir/train'
vaihingenDSMPath = tempRootPath/'vaihingen/dsm_dir/train'

# %%
rgbFiles = get_image_files(vaihingenRGBPath)
dsmFiles = get_image_files(vaihingenDSMPath)
# %%
import cv2
def show_dsm_image(temppath):
    tempImage = mmcv.imread(temppath,flag=-1)
    print(tempImage)
    show_image(tempImage, cmap = 'gray', figsize = (10, 10))
    return tempImage
tempImage1 = show_dsm_image(dsmFiles[0])

def show_rgb_image(temppath):
    tempImage = mmcv.imread(temppath)
    print(tempImage)
    show_image(tempImage, figsize = (10, 10))
    return tempImage    
tempImage2 = show_rgb_image(rgbFiles[0])
# %%
# using alpha channel for storing depth information
tempImage2.shape

# %%

def getFusionImage(tempImage1,tempImage2):
    tempImage3 = np.expand_dims(tempImage1,axis=2)
    alpha = np.repeat(np.arange(512,dtype=np.uint8)[np.newaxis,:], 512, axis=0)

    fuionImage = np.concatenate((tempImage2,tempImage3/1000),axis=2)
    # fuionImage = np.concatenate((tempImage2,np.expand_dims(alpha,axis=2)),axis=2)
    # plt.imshow(temp)
    show_image(fuionImage[:,:,:3].astype(np.uint8))
    return fuionImage
temp = getFusionImage(tempImage1,tempImage2)


# %%
# turn a float image to uint8
def normalize8(I):
  mn = I.min()
  mx = I.max()

  mx -= mn

  I = ((I - mn)/mx) * 255
  return np.round(I).astype(np.uint8)
# %%
