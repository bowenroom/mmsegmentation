# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset
# from collections import OrderedDict
# from functools import reduce

# import mmcv
# import numpy as np
# from mmcv.utils import print_log
# from prettytable import PrettyTable

# from mmseg.core import eval_metrics, pre_eval_to_metrics


@DATASETS.register_module()
class SEULCCDataset(BaseSegDataset):
    """seu land cover classification dataset.

    ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.png'.
    """

    METAINFO = dict(

    # order of the palette need to be consistent with the order of the label in the tools/convert_datasets/vaihingen.py and potsdam.py
    classes = ("BACKGROUND", "WATER", "CAR", "BUILDING", "PLAYGROUND"),
    # palette is in RGB order

    palette = [
    (0, 0, 0),
    (0, 0, 128),
    (0, 128, 0),
    (128, 0, 0),
    (128, 0, 128)

    ]
    )
    # 当不采用ignore的时候，模型性能有大幅提升
    def __init__(self, **kwargs):
        super(SEULCCDataset, self).__init__(
            img_suffix=".tif",
            seg_map_suffix=".tif",
            # reduce_zero_label=True,
            # ignore_index=1,
            **kwargs
        )
