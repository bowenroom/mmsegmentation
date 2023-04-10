# Copyright (c) OpenMMLab. All rights reserved.
<<<<<<< HEAD
from .builder import DATASETS
from .custom import CustomDataset
from collections import OrderedDict
from functools import reduce
=======
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset
>>>>>>> upstream/main

import mmcv
import numpy as np
from mmcv.utils import print_log
from prettytable import PrettyTable

from mmseg.core import eval_metrics, pre_eval_to_metrics

@DATASETS.register_module()
class ISPRSDataset(BaseSegDataset):
    """ISPRS dataset.

    In segmentation map annotation for ISPRS, 0 is the ignore index.
    ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.png'.
    """
<<<<<<< HEAD
    # order of the palette need to be consistent with the order of the label in the tools/convert_datasets/vaihingen.py and potsdam.py
    # CLASSES = ('impervious_surface', 'building', 'low_vegetation', 'tree',
    #            'car', 'clutter')
    CLASSES = ('impervious_surface', 'clutter', 'car', 'tree',
               'low_vegetation', 'building')

    # PALETTE = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0],
    #            [255, 255, 0], [255, 0, 0]]
    PALETTE = [[255, 255, 255], [255, 0, 0], [255, 255, 0],
               [0, 255, 0], [0, 255, 255], [0, 0, 255]]
# 当不采用ignore的时候，模型性能有大幅提升
    def __init__(self, **kwargs):
        super(ISPRSDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=True,
            # ignore_index=1,
=======
    METAINFO = dict(
        classes=('impervious_surface', 'building', 'low_vegetation', 'tree',
                 'car', 'clutter'),
        palette=[[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0],
                 [255, 255, 0], [255, 0, 0]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
>>>>>>> upstream/main
            **kwargs)

