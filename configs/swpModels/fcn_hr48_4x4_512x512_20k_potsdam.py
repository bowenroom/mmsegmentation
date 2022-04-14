_base_ = [
    '../_base_/models/fcn_hr18.py', '../_base_/datasets/potsdam.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
evaluation = dict(interval=288, metric='mIoU', pre_eval=True, save_best='mIoU')
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='WandbLoggerHook', init_kwargs=dict(project='Potsdam'))
    ])
data = dict(samples_per_gpu=7, workers_per_gpu=10)
checkpoint_config = dict(interval=300, save_optimizer=True, max_keep_ckpts=2)
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w48',
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384)))),
    decode_head=dict(
        in_channels=[48, 96, 192, 384], channels=sum([48, 96, 192, 384]),num_classes=6))
