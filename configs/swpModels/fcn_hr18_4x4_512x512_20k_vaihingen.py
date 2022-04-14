_base_ = [
    '../_base_/models/fcn_hr18.py', '../_base_/datasets/vaihingen.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
evaluation = dict(interval=288, metric='mIoU', pre_eval=True, save_best='mIoU')
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
        dict(type='WandbLoggerHook', init_kwargs=dict(project='Vaihingen'))
    ])
data = dict(samples_per_gpu=10, workers_per_gpu=10)
checkpoint_config = dict(interval=300, save_optimizer=True, max_keep_ckpts=3)
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(decode_head=dict(num_classes=6))
# model = dict(decode_head=dict(
#     _delete_ = True,
#         type='SegformerHead',
#         in_channels=[18, 36, 72, 144],
#         in_index=[0, 1, 2, 3],
#         channels=270,
#         # input_transform='resize_concat',
#         dropout_ratio=0.1,
#         num_classes=6,
#         norm_cfg=norm_cfg,
#         align_corners=False,
#         loss_decode=dict(
#             type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)))

