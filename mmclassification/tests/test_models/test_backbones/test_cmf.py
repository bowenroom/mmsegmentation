#%%
import torch
from mmseg.models.backbones import CMF
#%%
def test_convnext():
    model = CMF(arch='tiny',out_indices=-1)
    model.init_weights()
print('are u ok')