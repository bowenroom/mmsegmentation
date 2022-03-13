
_base_ = ['./segformer_mit-b1_512x512_40k_vaihingen.py'
          ]

# model settings

model = dict(
    pretrained='https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b4_8x1_1024x1024_160k_cityscapes/segformer_mit-b4_8x1_1024x1024_160k_cityscapes_20211207_080709-07f6c333.pth',
    backbone=dict(
        embed_dims=64, num_heads=[1, 2, 5, 8], num_layers=[3, 8, 27, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512], num_classes=6) 
)
data = dict(samples_per_gpu=4, workers_per_gpu=10)

