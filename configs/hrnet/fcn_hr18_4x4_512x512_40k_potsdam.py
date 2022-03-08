_base_ = [
    '../_base_/models/fcn_hr18.py', '../_base_/datasets/potsdam.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
evaluation = dict(interval=288, metric='mIoU', pre_eval=True, save_best='mIoU')
model = dict(decode_head=dict(num_classes=6))
