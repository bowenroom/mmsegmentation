# Copyright (c) OpenMMLab. All rights reserved.
#%%
import os
import os.path as osp
import shutil
import tempfile
from typing import Generator
from unittest.mock import MagicMock, patch

import numpy as np
# import pytest
import torch
from PIL import Image
from mmseg.datasets import (DATASETS, ADE20KDataset, CityscapesDataset,
                            COCOStuffDataset, ConcatDataset, CustomDataset,
                            ISPRSDataset, LoveDADataset, MultiImageMixDataset,
                            PascalVOCDataset, PotsdamDataset, RepeatDataset,FusionCustomDataset,
                            build_dataset, iSAIDDataset)


data_root = osp.join(osp.dirname(__file__), '../../swpTest/tempDataTest/vaihingen')
img_dir = 'img_dir/train'
ann_dir = 'ann_dir/train'
dsm_dir = 'dsm_dir_3ch/train'

def test_fusion_dataset():
    dataRoot = '/home/swp/paperCode/IJAGCode/swpTest/tempDataTest/vaihingen/' 
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    crop_size = (512, 512)
    train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', reduce_zero_label=True),
        dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 2.0)),
        dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
        dict(type='RandomFlip', prob=0.5),
        dict(type='PhotoMetricDistortion'),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg']),
    ]
    test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(512, 512),
            # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img']),
            ])
    ]

    # with img_dir and ann_dir
    train_dataset = FusionCustomDataset(
        pipeline=train_pipeline,
        # img_dir='img_dir/train',
        img_dir=img_dir,
        data_root=dataRoot,
        ann_dir= ann_dir,
        img_suffix='.png',
        seg_map_suffix='.png',
        multi_modality = True,
        dsm_suffix= 'tiff',
        dsm_dir=dsm_dir
        )
    # assert len(train_dataset) == 5
test_fusion_dataset()
# %%
def test_potsdam():
    test_dataset = PotsdamDataset(
        pipeline=[],
        img_dir=osp.join(
            osp.dirname(__file__), '../data/pseudo_potsdam_dataset/img_dir'),
        ann_dir=osp.join(
            osp.dirname(__file__), '../data/pseudo_potsdam_dataset/ann_dir'))
    assert len(test_dataset) == 1
    return test_dataset

test_potsdam()

# %%
