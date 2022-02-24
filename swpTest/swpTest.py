# test for different modules
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


# %%
# understand the parser


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert potsdam dataset to mmsegmentation format')
    parser.add_argument('dataset_path', help='potsdam folder path')
    parser.add_argument('--tmp_dir', help='path of the temporary directory')
    parser.add_argument('-o', '--out_dir', help='output path')
    parser.add_argument(
        '--clip_size',
        type=int,
        help='clipped size of image after preparation',
        default=512)
    parser.add_argument(
        '--stride_size',
        type=int,
        help='stride of clipping original images',
        default=256)
    # set the default break point
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print(args)
    import gc
    gc.collect()


# if __name__ == '__main__':
#     main()
# %%
# read an original label image
set_seed(888)
train_img_path = Path('../data/vaihingen/img_dir/train')
train_gt_path = Path('../data/vaihingen/ann_dir/train')
rgbFiles = get_image_files(train_img_path)
gtFiles = get_image_files(train_gt_path)
# %%
# read the label image
# gtImage = mmcv.imread(gtFiles[random.randint(0,100)],flag='grayscale',channel_order='rgb')


def colormap():
    #  #FFFFFF #0000FF #00FFFF #00FF00 #FFFF00 #FF0000
    # cdict = ['#FFFFFF', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00']
    cdict = ['#000000', '#FFFFFF', '#FF0000',
             '#FFFF00',  '#00FF00', '#00FFFF', '#0000FF']
    # 按照上面定义的colordict，将数据分成对应的部分，indexed：代表顺序
    return colors.ListedColormap(cdict, 'from_list')
# define my own pixel color paletter in the matplotlib
my_cmap = colormap()

# palette for vaihingen dataset
palette = {
    0:  (0, 0, 0),  # Undefined (black)
    1:  (255, 255, 255),  # Impervious surfaces (white)
    2:  (255, 0, 0),     # Clutter (red)
    3:  (255, 255, 0),   # Cars (yellow)
    4: (0, 255, 0),     # Trees (green)
    5: (0, 255, 255),   # Low vegetation (cyan)
    6: (0, 0, 255)     # Buildings (blue)
}
invert_palette = {v: k for k, v in palette.items()}
paletteValue = list(palette.values())


for i in range(8):
    temp = gtFiles[random.randint(0, 100)]
    # temp = gtFiles[0]
    print(temp)
    gtImage = mmcv.imread(temp, flag='grayscale', channel_order='rgb')
    # fig_size = (5, 5)
    # plt.figure(figsize=fig_size)
    # plt.imshow(gtImage, cmap=my_cmap)

    show_image(gtImage, cmap=my_cmap)
    # if len(np.unique(gtImage))<8:
    #     print(temp)
    #     show_image(gtImage, cmap=my_cmap)

# %%
# 原始代码中channel顺序没有一一对应，需要修改channel顺序，重新进行crop
# try to crop the original image manually and show the image result
originalFiles = get_image_files(Path(
    '/home/ubuntu/rsData/kaggleOriginal/Vaihingen/gts_eroded_for_participants/'))

image = mmcv.imread(originalFiles[0], channel_order='rgb')
h, w, c = image.shape
color_map = np.array([[0, 0, 0], [255, 255, 255], [255, 0, 0],
                      [255, 255, 0], [0, 255, 0], [0, 255, 255],
                      [0, 0, 255]])
flatten_v = np.matmul(
    image.reshape(-1, c),
    np.array([2, 3, 4]).reshape(3, 1))
out = np.zeros_like(flatten_v)
for idx, class_color in enumerate(color_map):
    value_idx = np.matmul(class_color,
                          np.array([2, 3, 4]).reshape(3, 1))
    out[flatten_v == value_idx] = idx
image = out.reshape(h, w)
# mmcv.imwrite(
#     image.astype(np.uint8),
#     '/home/ubuntu/paperCode/codeLib/mmsegmentation/swpTestImage/test.png')
testImage = mmcv.imread('/home/ubuntu/paperCode/codeLib/mmsegmentation/swpTest/swpTestImage/test.png',flag='grayscale')
# label is right now
show_image(testImage, cmap=my_cmap, figsize=(15, 15))

# %%
# pick up some images for the testset

test = rgbFiles[random.randint(0, 100)]
testImage= mmcv.imread(test)
show_image(testImage)
# %%
# remove the corrupted image
