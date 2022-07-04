# dataset settings
dataset_type = 'ISPRSDataset'
# data_root = 'data/vaihingen'
data_root = 'tempDataset/vaihingen'
# img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[
#                     58.395, 57.12, 57.375], to_rgb=True)
# img_norm_cfg = dict(
#     mean=[120.476, 81.7993, 81.1927,],
#     std=[54.8465, 39.3214, 37.9183],
#     to_rgb=True)


# EDFT Norm
img_norm_cfg = dict(
    mean=[120.476, 81.7993, 81.1927, 30.672],
    std=[54.8465, 39.3214, 37.9183, 38.0866],
    to_rgb=False)


# ImgeNet Norm
# (17568.pth,using the following norm cfg)
# img_norm_cfg = dict(mean=[123.675, 116.28, 103.53, 30.672], std=[
#                     58.395, 57.12, 57.375, 38.0866], to_rgb=True)


crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile', color_type='unchanged'),
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
    dict(type='LoadImageFromFile', color_type='unchanged'),
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
data = dict(
    samples_per_gpu=10,
    workers_per_gpu=10,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        # img_dir='img_dir/train',
        img_dir='img_dsm_dir/train',
        ann_dir='ann_dir/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        # img_dir='img_dir/val',
        img_dir='img_dsm_dir/val',
        ann_dir='ann_dir/val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dsm_dir/testFog',
        ann_dir='ann_dir/testFog',
        # img_dir='img_dir/val',
        # ann_dir='ann_dir/val',
        pipeline=test_pipeline))
