# Copyright (c) OpenMMLab. All rights reserved.
# %%
import argparse
import glob
import math
import os
import os.path as osp
import tempfile
import zipfile

import mmcv
import numpy as np

import matplotlib.colors as colors
import mmcv
import numpy as np
# import  fastai related modules
from fastai.basics import *
from fastai.vision.all import *

#%%
def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert vaihingen dataset to mmsegmentation format')
    parser.add_argument('dataset_path', help='vaihingen folder path')
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
    parser.add_argument(
        '--clip_size',
        type=int,
        help='clipped size of image after preparation',
        default=512)
    
    args = parser.parse_args()
    return args


def clip_big_image(image_path, clip_save_dir, to_label=False,dsm=False):
    # Original image of Vaihingen dataset is very large, thus pre-processing
    # of them is adopted. Given fixed clip size and stride size to generate
    # clipped image, the intersection　of width and height is determined.
    # For example, given one 5120 x 5120 original image, the clip size is
    # 512 and stride size is 256, thus it would generate 20x20 = 400 images
    # whose size are all 512x512.
    if dsm:
        image = mmcv.imread(image_path,flag=-1)
        h, w = image.shape
    else:
        # channel order is required to be correspondent with the palette order
        image = mmcv.imread(image_path, channel_order='rgb')
        h, w, c = image.shape
        
    
    

    
    cs = args.clip_size
    ss = args.stride_size

    num_rows = math.ceil((h - cs) / ss) if math.ceil(
        (h - cs) / ss) * ss + cs >= h else math.ceil((h - cs) / ss) + 1
    num_cols = math.ceil((w - cs) / ss) if math.ceil(
        (w - cs) / ss) * ss + cs >= w else math.ceil((w - cs) / ss) + 1

    x, y = np.meshgrid(np.arange(num_cols + 1), np.arange(num_rows + 1))
    xmin = x * cs
    ymin = y * cs

    xmin = xmin.ravel()
    ymin = ymin.ravel()
    xmin_offset = np.where(xmin + cs > w, w - xmin - cs, np.zeros_like(xmin))
    ymin_offset = np.where(ymin + cs > h, h - ymin - cs, np.zeros_like(ymin))
    boxes = np.stack([
        xmin + xmin_offset, ymin + ymin_offset,
        np.minimum(xmin + cs, w),
        np.minimum(ymin + cs, h)
    ],
        axis=1)

    if to_label:
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

    for box in boxes:
        start_x, start_y, end_x, end_y = box
        if dsm:
            clipped_image = image[start_y:end_y, start_x:end_x]
        else:
            clipped_image = image[start_y:end_y,
                                start_x:end_x] if to_label else image[
                                    start_y:end_y, start_x:end_x, :]

        area_idx = osp.basename(image_path).split('_')[3].strip('.tif')
        if dsm:
            mmcv.imwrite(
                clipped_image,
                osp.join(clip_save_dir,
                        f'{area_idx}_{start_x}_{start_y}_{end_x}_{end_y}.tiff'))
        else:
            mmcv.imwrite(
                clipped_image.astype(np.uint8),
                osp.join(clip_save_dir,
                        f'{area_idx}_{start_x}_{start_y}_{end_x}_{end_y}.png'))


def main():
    splits = {
        'train': [
            'area1',  'area3',  'area5'
        ],
        'val': [
            'area7'
        ],
        'test': ['area11']
    }

    dataset_path = args.dataset_path
    if args.out_dir is None:
        out_dir = osp.join('data', 'vaihingen')
    else:
        out_dir = args.out_dir

    print('Making directories...')
    mmcv.mkdir_or_exist(osp.join(out_dir, 'img_dir', 'train'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'img_dir', 'val'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'img_dir', 'test'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'train'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'val'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'test'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'dsm_dir', 'train'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'dsm_dir', 'val'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'dsm_dir', 'test'))


    zipp_list = glob.glob(os.path.join(dataset_path, '*.zip'))
    print('Find the data', zipp_list)

    with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmp_dir:
        for zipp in zipp_list:
            zip_file = zipfile.ZipFile(zipp)
            zip_file.extractall(tmp_dir)
            src_path_list = glob.glob(os.path.join(tmp_dir, '*.tif'))
                # delete unused area9 ground truth
            for area_ann in src_path_list:
                if 'area9' in area_ann:
                    src_path_list.remove(area_ann)
            prog_bar = mmcv.ProgressBar(len(src_path_list))
            for i, src_path in enumerate(src_path_list):
                area_idx = osp.basename(src_path).split('_')[3].strip('.tif')
                if area_idx in splits['train']:
                    data_type = 'train'
                elif area_idx in splits['val']:
                    data_type = 'val'
                elif area_idx in splits['test']:
                    data_type = 'test'
                if 'noBoundary' in src_path:
                    dst_dir = osp.join(out_dir, 'ann_dir', data_type)
                    clip_big_image(src_path, dst_dir, to_label=True)
                elif 'matching' in src_path:
                    dst_dir = osp.join(out_dir, 'dsm_dir', data_type)
                    # if to save the dsm files, we need to read the image using flag '-1' and write the image using tiff files 
                    clip_big_image(src_path, dst_dir,dsm=True)
                else:
                    dst_dir = osp.join(out_dir, 'img_dir', data_type)
                    clip_big_image(src_path, dst_dir, to_label=False)
                prog_bar.update()

        print('Removing the temporary files...')

    print('Done!')


if __name__ == '__main__':
    args = parse_args()
    main()
# %%
# validatae the image
# dsm_path = Path('/home/ubuntu/rsData/kaggleOriginal/Vaihingen/dsm')
dsm_path = Path('/home/ubuntu/paperCode/codeLib/mmsegmentation/swpTest/tempDataTest/dsm_dir')
dsmFiles= get_image_files(dsm_path)
dsmFiles[5]
dsmImage = mmcv.imread(dsmFiles[0],flag=-1)
# mmcv.imwrite(dsmImage, '/home/ubuntu/paperCode/codeLib/mmsegmentation/swpTest/swpTestImage/dsmTest.tiff')
# dsmImage2 = mmcv.imread('/home/ubuntu/paperCode/codeLib/mmsegmentation/swpTest/swpTestImage/dsmTest.tiff',flag=-1)
# np.unique(dsmImage2)
# test_eq(np.unique(dsmImage2),np.unique(dsmImage))
show_image(dsmImage,cmap='gray')
# %%
img_path = Path('/home/ubuntu/paperCode/codeLib/mmsegmentation/swpTest/tempDataTest/img_dir/train')
gt_path = Path('/home/ubuntu/paperCode/codeLib/mmsegmentation/swpTest/tempDataTest/ann_dir/train')
imgFiles= get_image_files(img_path)
gtFiles= get_image_files(gt_path)
imgFiles[5]
imgImage = mmcv.imread(imgFiles[0])
gtImage = mmcv.imread(gtFiles[0],flag=-1)
def colormap():
    #  #FFFFFF #0000FF #00FFFF #00FF00 #FFFF00 #FF0000
    # cdict = ['#FFFFFF', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00']
    cdict = ['#000000', '#FFFFFF', '#FF0000',
             '#FFFF00',  '#00FF00', '#00FFFF', '#0000FF']
    # 按照上面定义的colordict，将数据分成对应的部分，indexed：代表顺序
    return colors.ListedColormap(cdict, 'from_list')
# define my own pixel color paletter in the matplotlib
my_cmap = colormap()

# %%
def show_test(n):
    for i in range(n):
        imgImage = mmcv.imread(imgFiles[i])
        gtImage = mmcv.imread(gtFiles[i],flag=-1)
        show_image(imgImage)
        show_image(gtImage,cmap=my_cmap)
show_test(10)
# %%
