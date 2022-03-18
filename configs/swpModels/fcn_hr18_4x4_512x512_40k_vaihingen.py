_base_ = [
    '../_base_/models/fcn_hr18.py', '../_base_/datasets/vaihingen.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
evaluation = dict(interval=288, metric='mIoU', pre_eval=True, save_best='mIoU')
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
        dict(type='WandbLoggerHook', init_kwargs=dict(project='IGRL'))
    ])
checkpoint_config = dict(interval=300, save_optimizer=True, max_keep_ckpts=3)
model = dict(decode_head=dict(num_classes=6))
data = dict(samples_per_gpu=10, workers_per_gpu=10)
