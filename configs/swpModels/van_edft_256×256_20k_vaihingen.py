_base_ = [
    '../_base_/models/segformer_mit-b0.py',
    '../_base_/datasets/isprsMultiMod.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

model = dict(
    # pretrained='pretrain/mit_fuse_b0.pth',
    backbone=dict(
        type='WPLKA',
        backbone="Segformer",
        in_channels=4,
        weight=0.5,
        overlap=True,
        attention_type='dsa-add',
        same_branch=False),
    decode_head=dict(num_classes=6))

# log
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
        dict(type='WandbLoggerHook', init_kwargs=dict(project='MultiModality'))
    ])
# optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW',
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

checkpoint_config = dict(interval=300, save_optimizer=True, max_keep_ckpts=2)
data = dict(samples_per_gpu=8, workers_per_gpu=0)
evaluation = dict(interval=288,metric=['mIoU', 'mFscore'], save_best='mIoU')
