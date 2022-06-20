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
import re
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
        image = mmcv.imread(image_path, channel_order='rgb',flag='unchanged')
        h, w, c = image.shape
        assert c==4
        
    
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
        color_map = np.array([[0, 0, 0],[255, 255, 255], [255, 0, 0],
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
                        f'{area_idx}_{start_x}_{start_y}_{end_x}_{end_y}.png'))
        else:
            mmcv.imwrite(
                clipped_image.astype(np.uint8),
                osp.join(clip_save_dir,
                        f'{area_idx}_{start_x}_{start_y}_{end_x}_{end_y}.tif'))

def clip_big_image_pot(image_path, clip_save_dir, to_label=False,dsm=False):
    # Original image of Potsdam dataset is very large, thus pre-processing
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

    clip_size = args.clip_size
    stride_size = args.stride_size

    num_rows = math.ceil((h - clip_size) / stride_size) if math.ceil(
        (h - clip_size) /
        stride_size) * stride_size + clip_size >= h else math.ceil(
            (h - clip_size) / stride_size) + 1
    num_cols = math.ceil((w - clip_size) / stride_size) if math.ceil(
        (w - clip_size) /
        stride_size) * stride_size + clip_size >= w else math.ceil(
            (w - clip_size) / stride_size) + 1

    x, y = np.meshgrid(np.arange(num_cols + 1), np.arange(num_rows + 1))
    xmin = x * clip_size
    ymin = y * clip_size

    xmin = xmin.ravel()
    ymin = ymin.ravel()
    xmin_offset = np.where(xmin + clip_size > w, w - xmin - clip_size,
                           np.zeros_like(xmin))
    ymin_offset = np.where(ymin + clip_size > h, h - ymin - clip_size,
                           np.zeros_like(ymin))
    boxes = np.stack([
        xmin + xmin_offset, ymin + ymin_offset,
        np.minimum(xmin + clip_size, w),
        np.minimum(ymin + clip_size, h)
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
        idx_i, idx_j = osp.basename(image_path).split('_')[2:4]
        if 'dsm' in image_path:
            idx_i = re.sub('^0*', '', idx_i)
            idx_j = idx_j.strip(".tif")
            idx_j = re.sub('^0*', '', idx_j)
        if dsm:
             mmcv.imwrite(
                clipped_image,
                osp.join(clip_save_dir,
                        f'{idx_i}_{idx_j}_{start_x}_{start_y}_{end_x}_{end_y}.tiff'))
        else:
            mmcv.imwrite(
                clipped_image.astype(np.uint8),
                osp.join(
                    clip_save_dir,
                    f'{idx_i}_{idx_j}_{start_x}_{start_y}_{end_x}_{end_y}.png'))



def main():
    # Vaihingen
    splits = {
        'train': [
            'area1', 'area11', 'area13', 'area15', 'area17', 'area21',
            'area23', 'area26', 'area28', 'area3', 'area30', 'area32',
            'area34', 'area37', 'area5', 'area7','area6', 'area24',
            'area35', 'area16', 'area14', 'area22','area10', 'area4',
            'area2', 'area20', 'area8', 'area31', 'area33', 'area29'
    
        ],
        'val': [
            
            'area27', 'area38', 'area12'
        ],
    }
    # Potsdam
    # splits = {
    #     'train': [
    #         '2_10', '2_11', '2_12', '3_10', '3_11', '3_12', '4_10', '4_11',
    #          '5_10', '5_11', '5_12', '6_10', '6_11', '6_12', '6_7',
    #         '6_8', '6_9', '7_10', '7_11', '7_12', '7_7', '7_8', '7_9',
    #         '5_15', '6_15', '6_13','3_13','4_14', '6_14', '5_14', '2_13','3_14', '7_13','4_13'
    #     ],
    #     'val': [
            
    #         '4_15', '2_14', '5_13' 
    #     ]
    # }

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


    zipp_list = glob.glob(os.path.join(dataset_path, '*.zip'))
    print('Find the data', zipp_list)

    with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmp_dir:
        for zipp in zipp_list:
            zip_file = zipfile.ZipFile(zipp)
            zip_file.extractall(tmp_dir)
            # src_path_list = glob.glob(os.path.join(tmp_dir, '*.tif'))
                # delete unused area9 ground truth
            if 'vaihingenImage' in zipp:
                src_path_list =  glob.glob(os.path.join(os.path.join(tmp_dir, '**'),'*.tif'))
                for area_ann in src_path_list:
                    if 'area9' in area_ann:
                        src_path_list.remove(area_ann)
            if 'vaihingenAnn' in zipp:  # noqa
                src_path_list = glob.glob(os.path.join(os.path.join(tmp_dir, '**'),'*.png'))
                # delete unused area9 ground truth
                for area_ann in src_path_list:
                    if 'area9' in area_ann:
                        src_path_list.remove(area_ann)
            if '2_Ortho_RGB' in zipp:
                src_path_list = glob.glob(os.path.join(tmp_dir, '*.tif'))
                for area_ann in src_path_list:
                    if '4_12' in area_ann:
                        src_path_list.remove(area_ann)
            prog_bar = mmcv.ProgressBar(len(src_path_list))
            if 'vaihingen' in out_dir:
                for i, src_path in enumerate(src_path_list):
                    area_idx = osp.basename(src_path).split('_')[3].strip('.tif')
                    if area_idx in splits['train']:
                        data_type = 'train'
                    elif area_idx in splits['val']:
                        data_type = 'val'
                    if 'noBoundary' in src_path:
                        dst_dir = osp.join(out_dir, 'ann_dir', data_type)
                        clip_big_image(src_path, dst_dir, dsm=True)
                    else:
                        dst_dir = osp.join(out_dir, 'img_dir', data_type)
                        clip_big_image(src_path, dst_dir, to_label=False)
                prog_bar.update()
            elif 'potsdam' in out_dir:
                for i, src_path in enumerate(src_path_list):
                    idx_i, idx_j = osp.basename(src_path).split('_')[2:4]
                    if 'dsm' in src_path:
                        idx_i = re.sub('^0*', '', idx_i)
                        idx_j = idx_j.strip(".tif")
                        idx_j = re.sub('^0*', '', idx_j)
                    data_type = 'train' if f'{idx_i}_{idx_j}' in splits[
                        'train'] else 'val'
                    if 'label' in src_path:
                        dst_dir = osp.join(out_dir, 'ann_dir', data_type)
                        clip_big_image_pot(src_path, dst_dir, to_label=True)
                    elif 'RGB' in src_path:
                        dst_dir = osp.join(out_dir, 'img_dir', data_type)
                        clip_big_image_pot(src_path, dst_dir, to_label=False)
                    elif 'dsm' in src_path:
                        dst_dir = osp.join(out_dir, 'dsm_dir', data_type)
                        clip_big_image_pot(src_path, dst_dir, dsm=True)
                    prog_bar.update()                

        print('Removing the temporary files...')

    print('Done!')


if __name__ == '__main__':
    args = parse_args()
    main()
# %%
'''
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
# show_test(10)
# %%
'''