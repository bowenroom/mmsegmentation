# %%
from scoring import score_masks
from mmseg.core.evaluation import mean_fscore
from mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from PIL import Image
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import mmcv
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmseg.apis.inference import init_segmentor

from mmseg.datasets.pipelines import Compose
from mmseg.models import build_segmentor

from config import train_ids, test_ids, val_ids, LABELMAP_RGB

# LABELMAP_RGB:
# {0: (255, 0, 255), 1: (230, 25, 75), 2: (145, 30, 180), 3: (60, 180, 75), 4: (245, 130, 48), 5: (255, 255, 255), 6: (0, 130, 200)}


def category2mask(img):
    """ Convert a category image to color mask """
    if len(img) == 3:
        if img.shape[2] == 3:
            img = img[:, :, 0]

    mask = np.zeros(img.shape[:2] + (3, ), dtype='uint8')

    for category, mask_color in LABELMAP_RGB.items():
        locs = np.where(img == category)
        mask[locs] = mask_color

    return mask


def chips_from_image(img, size=300):
    shape = img.shape

    chip_count = math.ceil(shape[1] / size) * math.ceil(shape[0] / size)

    chips = []
    for x in range(0, shape[1], size):
        for y in range(0, shape[0], size):
            chip = img[y:y+size, x:x+size, :]
            y_pad = size - chip.shape[0]
            x_pad = size - chip.shape[1]
            chip = np.pad(chip, [(0, y_pad), (0, x_pad),
                          (0, 0)], mode='constant')
            chips.append((chip, x, y))
    return chips


def run_inference_on_file(imagefile, predsfile, model, size=300):
    with Image.open(imagefile).convert('RGB') as img:
        nimg = np.array(Image.open(imagefile).convert('RGB'))
        shape = nimg.shape
        chips = chips_from_image(nimg)

    chips = [(chip, xi, yi) for chip, xi, yi in chips if chip.sum() > 0]
    prediction = np.zeros(shape[:2], dtype='uint8')
    chip_preds = []
    for chip, _, _ in chips:
        temp = inference_segmentor(model, chip)
        chip_preds.extend(temp)

    # chip_preds = model.predict(
    #     np.array([chip for chip, _, _ in chips]), verbose=True)

    for (chip, x, y), pred in zip(chips, chip_preds):
        # category_chip = np.argmax(pred, axis=-1) + 1
        pred = pred + 1
        section = prediction[y:y+size, x:x+size].shape

        prediction[y:y+size, x:x +
                   size] = pred[:section[0], :section[1]]

    mask = category2mask(prediction)
    Image.fromarray(mask).save(predsfile)


def run_inference(dataset, model=None, model_path=None, basedir='predictions'):
    pred_path = os.path.join(dataset, basedir)
    if not os.path.isdir(pred_path):
        os.mkdir(pred_path)
    if model is None and model_path is None:
        raise Exception("model or model_path required")
    for scene in train_ids + val_ids + test_ids:
        # 作者直接在大图上进行inference，然后把各个子结果组装起来
        imagefile = f'{dataset}/images/{scene}-ortho.tif'
        predsfile = os.path.join(pred_path, f'{scene}-prediction.png')

        if not os.path.exists(imagefile):
            continue

        print(f'running inference on image {imagefile}.')
        run_inference_on_file(imagefile, predsfile, model)


# %%
# 开始inference
# 在大图上做inference

# img_path = '/home/swp/paperCode/IGRLCode/mmf/rsData/dronedeploy/images/1d4fbe33f3_F1BE1D4184INSPIRE-ortho.tif'
# lbl_path = '/home/swp/paperCode/IGRLCode/mmf/rsData/dronedeploy/labels/1d4fbe33f3_F1BE1D4184INSPIRE-label.png'
img_path = '/home/swp/paperCode/IGRLCode/mmf/tempDataset/ddsb2/img_dir/test/1af86939f_F1BE1D4184OPENPIPELINE_300_300_600_600.png'
lbl_path = img_path.replace('img_dir', 'ann_dir')
dataset_path = '/home/swp/paperCode/IGRLCode/mmf/rsData/dronedeploy'

img = mmcv.imread(img_path, channel_order='rgb')
config_file = '/home/swp/paperCode/IGRLCode/mmf/configs/swpModels/upernext_convnext_base_512x512_40k_ddsb.py'
checkpoint_file = '/home/swp/paperCode/IGRLCode/mmf/work_dirs/upernext_convnext_base_512x512_40k_ddsb/save/20221021_103757.pth'
model = init_segmentor(config_file, checkpoint_file, device='cuda:1')

# run_inference(dataset_path, model, checkpoint_file)
# result = inference_segmentor(model, img)
# show_result_pyplot(model, img, result,
#                    palette=get_palette('ddsb'), opacity=0.6)
# lbl = mmcv.imread(lbl_path, flag='grayscale')

# %%
# 测试一下原来的score
from scoring import score_predictions
score, conf = score_predictions('/home/swp/paperCode/IGRLCode/mmf/rsData/dronedeploy', basedir='predictions')
print(f'score is {score}')
