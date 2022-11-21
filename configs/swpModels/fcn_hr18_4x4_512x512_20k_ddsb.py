_base_ = [
    '../_base_/models/ocrnet_hr18.py', '../_base_/datasets/ddsb.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w48',
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384)))),
    decode_head=[
        dict(
            type='FCNHead',
            in_channels=[48, 96, 192, 384],
            channels=sum([48, 96, 192, 384]),
            input_transform='resize_concat',
            in_index=(0, 1, 2, 3),
            kernel_size=1,
            num_convs=1,
            norm_cfg=norm_cfg,
            concat_input=False,
            dropout_ratio=-1,
            num_classes=6,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='OCRHead',
            in_channels=[48, 96, 192, 384],
            channels=512,
            ocr_channels=256,
            input_transform='resize_concat',
            in_index=(0, 1, 2, 3),
            norm_cfg=norm_cfg,
            dropout_ratio=-1,
            num_classes=6,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
    ])
    
evaluation = dict(interval=1288, metric=['mIoU', 'mFscore'], pre_eval=True)
log_config = dict(
interval=50,
hooks=[
    dict(type='TextLoggerHook', by_epoch=False),
    dict(type='WandbLoggerHook', init_kwargs=dict(project='DDSB'))
])
data = dict(samples_per_gpu=5, workers_per_gpu=0)
checkpoint_config = dict(interval=644, save_optimizer=True, max_keep_ckpts=3)
norm_cfg = dict(type='SyncBN', requires_grad=True)
load_from = '/home/swp/paperCode/IGRLCode/mmf/work_dirs/fcn_hr18_4x4_512x512_20k_ddsb/iter_40000.pth'
