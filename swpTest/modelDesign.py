# test for different modules
# %%
# import packages
import argparse
import glob
import imp
import math
import os
import os.path as osp
import tempfile
import zipfile
import matplotlib.colors as colors
import mmcv
import numpy as np
# import  fastai related modules
from fastai.basics import *
from fastai.vision.all import *
# from pyrsistent import T
from timm.models.layers import create_attn


# %%
ecaAttn = create_attn('eca',3)
input = torch.randn(1,3,512,512)

out = ecaAttn(input)
# %%
