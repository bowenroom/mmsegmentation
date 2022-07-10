_base_ = [
    '../_base_/models/upernet_convnext.py', '../_base_/datasets/vaihingen.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
## data
crop_size = (512, 512)
# data = dict(samples_per_gpu=2, workers_per_gpu=2)
data = dict(samples_per_gpu=2, workers_per_gpu=0)


## model
model = dict(
    decode_head=dict(in_channels=[128, 256, 512, 1024], num_classes=150),
    auxiliary_head=dict(in_channels=512, num_classes=6),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(341, 341)),
)


## training

optimizer = dict(
    constructor='LearningRateDecayOptimizerConstructor',
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg={
        'decay_rate': 0.9,
        'decay_type': 'stage_wise',
        'num_layers': 12
    })

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

# fp16 settings
optimizer_config = dict(type='Fp16OptimizerHook', loss_scale='dynamic')
# fp16 placeholder
fp16 = dict()

norm_cfg = dict(type='SyncBN', requires_grad=True)

## evaluation and log
evaluation = dict(interval=1288, metric='mIoU',
                  pre_eval=True, save_best='mIoU')
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
        # dict(type='WandbLoggerHook', init_kwargs=dict(project='IJAG'))
    ])
checkpoint_config = dict(interval=644, save_optimizer=True, max_keep_ckpts=2)

# load_from = "pretrain/best_mIoU_iter_39284.pth"

# from mmclassification.mmcls.models import VAN