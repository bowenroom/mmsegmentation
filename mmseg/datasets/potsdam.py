# Copyright (c) OpenMMLab. All rights reserved.
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class PotsdamDataset(CustomDataset):
    """ISPRS Potsdam dataset.

    In segmentation map annotation for Potsdam dataset, 0 is the ignore index.
    ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.png'.
    """
    # order of the palette need to be consistent with the order of the label in the tools/convert_datasets/vaihingen.py and potsdam.py
    # CLASSES = ('impervious_surface', 'building', 'low_vegetation', 'tree',
    #            'car', 'clutter')
    CLASSES = ('impervious_surface', 'clutter', 'car', 'tree',
               'low_vegetation', 'building')

    # PALETTE = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0],
    #            [255, 255, 0], [255, 0, 0]]
    PALETTE = [[255, 255, 255], [255, 0, 0], [255, 255, 0], [0, 255, 0],
               [0, 255, 255], [0, 0, 255]]
    def __init__(self, **kwargs):
        super(PotsdamDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=True,
            ignore_index=1,
            **kwargs)
