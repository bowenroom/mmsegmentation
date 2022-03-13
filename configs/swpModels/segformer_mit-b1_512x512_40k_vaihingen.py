
_base_ = ['../_base_/models/segformer_mit-b0.py',
          '../_base_/datasets/vaihingen.py',
          '../_base_/default_runtime.py',
          '../_base_/schedules/schedule_40k.py'
          ]

# model settings

model = dict(
    pretrained='https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b1_8x1_1024x1024_160k_cityscapes/segformer_mit-b1_8x1_1024x1024_160k_cityscapes_20211208_064213-655c7b3f.pth',
    backbone=dict(
        embed_dims=64, num_heads=[1, 2, 5, 8], num_layers=[2, 2, 2, 2]),
    decode_head=dict(in_channels=[64, 128, 320, 512], num_classes=6),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(512, 512))
)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
        dict(type='WandbLoggerHook', init_kwargs=dict(project='IGRL'))
    ])
evaluation = dict(interval=288, metric='mIoU', pre_eval=True, save_best='mIoU')
# optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW',
    # lr=0.00006,
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
checkpoint_config = dict(interval=300, save_optimizer=True, max_keep_ckpts=3)
data = dict(samples_per_gpu=10, workers_per_gpu=10)
load_from = 'work_dirs/segformer_mit-b1_512x512_40k_vaihingen/best_mIoU_iter_22464.pth'
