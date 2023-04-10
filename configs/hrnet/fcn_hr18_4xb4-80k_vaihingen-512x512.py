_base_ = [
    '../_base_/models/fcn_hr18.py', '../_base_/datasets/vaihingen.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
<<<<<<< HEAD:configs/hrnet/fcn_hr18_4x4_512x512_80k_vaihingen.py
evaluation = dict(interval=288, metric='mIoU', pre_eval=True, save_best='mIoU')
model = dict(decode_head=dict(num_classes=6))
=======
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor, decode_head=dict(num_classes=6))
>>>>>>> upstream/main:configs/hrnet/fcn_hr18_4xb4-80k_vaihingen-512x512.py
