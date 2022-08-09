_base_ = [
    '../_base_/models/fcn_hr18.py', '../_base_/datasets/vaihingen.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
evaluation = dict(interval=1288, metric='mIoU', pre_eval=True, save_best='mIoU')
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
        dict(type='WandbLoggerHook', init_kwargs=dict(project='IJAG'))
    ])
data = dict(samples_per_gpu=2, workers_per_gpu=0)
checkpoint_config = dict(interval=644, save_optimizer=True, max_keep_ckpts=1)
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    backbone=dict(
        # multi = False,
        multi = True,
        # attention='LKA',
        weight=0.5,
        embed_dims=18, overlap=True, num_heads=[1, 2, 4, 8]),
    decode_head=dict(num_classes=6))

load_from = "pretrain/best_mIoU_iter_39284.pth"
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

