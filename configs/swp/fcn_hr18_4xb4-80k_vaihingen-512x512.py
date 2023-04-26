_base_ = [
    "../_base_/models/fcn_hr18.py",
    "../_base_/datasets/vaihingen.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_40k.py",
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor, decode_head=dict(num_classes=6))
train_cfg = dict(val_interval=1288)
checkpoint = dict(type="CheckpointHook", interval=1288,max_keep_ckpts=3)
vis_backends=[dict(type='LocalVisBackend'),
              dict(type='TensorboardVisBackend'),
            #   dict(type='WandbVisBackend')
              ]
visualizer = dict(
    type="SegLocalVisualizer",
    vis_backends=vis_backends,
    save_dir='visual',
    name="visualizer",
)
default_hooks = dict(
    visualization=dict(type='SegVisualizationHook',interval=1,draw=True))
