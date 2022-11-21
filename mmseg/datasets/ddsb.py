# Copyright (c) OpenMMLab. All rights reserved.
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class DDSBDataset(CustomDataset):
    """DroneDeploy Segmentation Benchmark dataset.

    ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.png'.
    """
    # order of the palette need to be consistent with the order of the label in the tools/convert_datasets/vaihingen.py and potsdam.py
    # CLASSES = ('impervious_surface', 'building', 'low_vegetation', 'tree',
    #            'car', 'clutter')
    CLASSES = ('BUILDING', 'CLUTTER', 'VEGETATION', 'WATER',
               'GROUND', 'CAR')

    PALETTE = [
        [230, 25, 75],
        [145, 30, 180],
        [60, 180, 75],
        [245, 130, 48],
        [255, 255, 255],
        [0, 130, 200]
    ]

    def __init__(self, **kwargs):
        super(DDSBDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            # reduce_zero_label=True,
            # ignore_index=1,
            **kwargs)
