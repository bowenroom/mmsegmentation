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


# %%
# understand the parser
def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert potsdam dataset to mmsegmentation format')
    parser.add_argument('dataset_path', help='potsdam folder path')
    parser.add_argument('--tmp_dir', help='path of the temporary directory')
    parser.add_argument('-o', '--out_dir', help='output path')
    parser.add_argument(
        '--clip_size',
        type=int,
        help='clipped size of image after preparation',
        default=512)
    parser.add_argument(
        '--stride_size',
        type=int,
        help='stride of clipping original images',
        default=256)
    # set the default break point
    args = parser.parse_args()
    return args
def main():
    args = parse_args()
    print(args)
    import gc
    gc.collect()

# if __name__ == '__main__':
#     main()
# %%
# understand the mmcv read and the palette
set_seed(888)
train_img_path = Path('../data/vaihingen/img_dir/train')
train_gt_path = Path('../data/vaihingen/ann_dir/train')
rgbFiles = get_image_files(train_img_path)
gtFiles = get_image_files(train_gt_path)
def colormap():
    #  #FFFFFF #0000FF #00FFFF #00FF00 #FFFF00 #FF0000
    # cdict = ['#FFFFFF', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00']
    # 
    cdict = ['#000000', '#FFFFFF', '#FF0000',
             '#FFFF00',  '#00FF00', '#00FFFF', '#0000FF']
    # 按照上面定义的colordict，将数据分成对应的部分，indexed：代表顺序
    return colors.ListedColormap(cdict, 'from_list')
# define my own pixel color paletter in the matplotlib
my_cmap = colormap()

# palette for vaihingen dataset
palette = {
    0:  (0, 0, 0),  # Undefined (black)
    1:  (255, 255, 255),  # Impervious surfaces (white)
    2:  (255, 0, 0),     # Clutter (red)
    3:  (255, 255, 0),   # Cars (yellow)
    4: (0, 255, 0),     # Trees (green)
    5: (0, 255, 255),   # Low vegetation (cyan)
    6: (0, 0, 255)     # Buildings (blue)
}

invert_palette = {v: k for k, v in palette.items()}
paletteValue = list(palette.values())

for i in range(8):
    temp = gtFiles[random.randint(0, 100)]
    # temp = gtFiles[0]
    print(temp)
    gtImage = mmcv.imread(temp, flag='grayscale', channel_order='rgb')
    # fig_size = (5, 5)
    # plt.figure(figsize=fig_size)
    # plt.imshow(gtImage, cmap=my_cmap)

    show_image(gtImage, cmap=my_cmap)
    # if len(np.unique(gtImage))<8:
    #     print(temp)
    #     show_image(gtImage, cmap=my_cmap)

# %%
# 原始代码中channel顺序没有一一对应，需要修改channel顺序，重新进行crop
# try to crop the original image manually and show the image result
originalFiles = get_image_files(Path(
    '/home/ubuntu/rsData/kaggleOriginal/Vaihingen/gts_eroded_for_participants/'))

image = mmcv.imread(originalFiles[0], channel_order='rgb')
h, w, c = image.shape
color_map = np.array([[0, 0, 0], [255, 255, 255], [255, 0, 0],
                      [255, 255, 0], [0, 255, 0], [0, 255, 255],
                      [0, 0, 255]])
flatten_v = np.matmul(
    image.reshape(-1, c),
    np.array([2, 3, 4]).reshape(3, 1))
out = np.zeros_like(flatten_v)
for idx, class_color in enumerate(color_map):
    value_idx = np.matmul(class_color,
                          np.array([2, 3, 4]).reshape(3, 1))
    out[flatten_v == value_idx] = idx
image = out.reshape(h, w)
# mmcv.imwrite(
#     image.astype(np.uint8),
#     '/home/ubuntu/paperCode/codeLib/mmsegmentation/swpTestImage/test.png')

# pick up some images for the testset
test = rgbFiles[random.randint(0, 100)]
testImage= mmcv.imread(test)
show_image(testImage)
# %%
#%%
# calculate the error between two dict
temp1 = {"aAcc": 0.8848, "mIoU": 0.7515, "mAcc": 0.8345, "IoU.impervious_surface": 0.8313, "IoU.building": 0.4054, "IoU.low_vegetation": 0.8851, "IoU.tree": 0.7545, "IoU.car": 0.7349, "IoU.clutter": 0.8975, "Acc.impervious_surface": 0.9431, "Acc.building": 0.5157, "Acc.low_vegetation": 0.9349, "Acc.tree": 0.8116, "Acc.car": 0.857, "Acc.clutter": 0.9451}
temp2 = {"aAcc": 0.8931, "mIoU": 0.7588, "mAcc": 0.8372, "IoU.impervious_surface": 0.8513, "IoU.building": 0.9123, "IoU.low_vegetation": 0.7379, "IoU.tree": 0.7774, "IoU.car": 0.8939, "IoU.clutter": 0.3801, "Acc.impervious_surface": 0.9093, "Acc.building": 0.964, "Acc.low_vegetation": 0.9086, "Acc.tree": 0.8509, "Acc.car": 0.9547, "Acc.clutter": 0.4356}

temp3 = {"aAcc": 0.9011, "mIoU": 0.7257, "mAcc": 0.8006, "IoU.impervious_surface": 0.8577, "IoU.building": 0.9163, "IoU.low_vegetation": 0.711, "IoU.tree": 0.7992, "IoU.car": 0.7424, "IoU.clutter": 0.3278, "Acc.impervious_surface": 0.9275, "Acc.building": 0.958, "Acc.low_vegetation": 0.8128, "Acc.tree": 0.9121, "Acc.car": 0.8608, "Acc.clutter": 0.3323}
temp4 ={"aAcc": 0.8863, "mIoU": 0.7022, "mAcc": 0.7797, "IoU.impervious_surface": 0.8307, "IoU.building": 0.9007, "IoU.low_vegetation": 0.6784, "IoU.tree": 0.7775, "IoU.car": 0.7278, "IoU.clutter": 0.2979, "Acc.impervious_surface": 0.9416, "Acc.building": 0.9347, "Acc.low_vegetation": 0.7873, "Acc.tree": 0.8831, "Acc.car": 0.8261, "Acc.clutter": 0.3055}
def calulate_error(gt_dict,pred_dict):
    error_dict = {}
    for key in gt_dict.keys():
        # calculate the error for each key
        error_dict[key] = abs(gt_dict[key]-pred_dict[key])
        # error_dict[key] = np.sum(np.abs(gt_dict[key]-pred_dict[key]))
    return error_dict
value = calulate_error(temp3,temp4)
# sort the dict,reverse
value = sorted(value.items(),key=lambda x:x[1],reverse=True)
value
#%%
# using wandb for log the value in the experiment
import wandb
config = dict (
  learning_rate = 0.01,
  momentum = 0.2,
  architecture = "CNN",
  dataset_id = "peds-0192",
  infra = "AWS",
)
# wandb.init(project="newTest",config=config)
#%%
tempPath = '/home/ubuntu/paperCode/codeLib/mmsegmentation/swpTest/tempDataTest/vaihingen/img_dir/val/area2_1916_2048_2428_2560.png'
# tempImage = mmcv.imread(tempPath, flag='grayscale', channel_order='rgb')
tempImage = mmcv.imread(tempPath)
show_image(tempImage, cmap=my_cmap,figsize = (10, 10)) 
# %%
def show_list_image():
    for i in range(5):
        tempPath = f'/home/ubuntu/paperCode/codeLib/mmsegmentation/swpTest/tempDataTest/vaihingen/img_dir/testFog/area2_1916_2048_2428_2560_fog_{i+1}.png'
        tempImage = mmcv.imread(tempPath)
        show_image(tempImage, cmap=my_cmap,figsize = (10, 10)) 
# show_list_image()
#%%
# show the dsm image 
dsmPath = get_image_files('/home/ubuntu/paperCode/codeLib/mmsegmentation/swpTest/tempDataTest/potsdam/dsm_dir/val')
dsmImage = dsmPath[random.randint(0,100)]
path = dsmImage
def show_dsm_image(temppath):
    tempImage = mmcv.imread(temppath,flag=-1)
    show_image(tempImage, cmap = 'gray', figsize = (10, 10))
show_dsm_image(path)
# %%
# %%
#/home/ubuntu/paperCode/codeLib/mmsegmentation/swpTest/tempDataTest/potsdam/dsm_dir/val/4_15_1024_1024_1536_1536.tiff
# show the image and the inference
from mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
imagePath = '/home/ubuntu/paperCode/codeLib/mmsegmentation/swpTest/tempDataTest/potsdam/img_dir/val/4_15_1024_1024_1536_1536.png'
config_file = '/home/ubuntu/paperCode/codeLib/mmsegmentation/configs/swpModels/fcn_hr18_4x4_512x512_40k_potsdam.py'
# checkpoint_file ='/home/ubuntu/paperCode/codeLib/mmsegmentation/work_dirs/fcn_hr18_4x4_512x512_40k_potsdam/best_mIoU_iter_39168_1.pth'
# checkpoint_file = '/home/ubuntu/paperCode/codeLib/mmsegmentation/work_dirs/fcn_hr18_4x4_512x512_40k_potsdam/iter_300.pth'
checkpoint_file = '/home/ubuntu/paperCode/codeLib/mmsegmentation/work_dirs/fcn_hr18_512x512_80k_potsdam/iter_8000.pth'
fogPath = osp.splitext(imagePath.replace('val','testFog'))[0]+'_fog_1'+'.png'
# fogPath = osp.splitext(imagePath.replace('val','testFog'))[0]+'_fog_3'+'.png'
def show_result(imagePath,config_file,checkpoint_file):
    axs = get_grid(5,1,5,figsize = (50, 10))
    image = mmcv.imread(imagePath)
    gtImage = mmcv.imread(imagePath.replace('img_dir','ann_dir'),flag='grayscale', channel_order='rgb')
    dsmImage = mmcv.imread(imagePath.replace('img_dir','dsm_dir').replace('.png','.tiff'),flag=-1)
    model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
    result = inference_segmentor(model, imagePath)
    result2 = inference_segmentor(model, fogPath)
    # fogImage = mmcv.imread(fogPath)
    inferImage =  model.show_result(
    imagePath, result, palette=get_palette('potsdam'), show=False, opacity=1)
    inferImage2 =  model.show_result(
        fogPath, result2, palette=get_palette('potsdam'), show=False, opacity=1)
    [i.set_axis_off() for i in axs]
    axs[0].imshow(image)
    axs[2].imshow(gtImage,cmap = my_cmap)
    axs[1].imshow(dsmImage,cmap = 'gray')
    axs[3].imshow(mmcv.bgr2rgb(inferImage))
    axs[4].imshow(mmcv.bgr2rgb(inferImage2))
show_result(imagePath,config_file,checkpoint_file)
#%%
def  inference_image():
    
    config_file = '/home/ubuntu/paperCode/codeLib/mmsegmentation/configs/swpModels/fcn_hr18_4x4_512x512_40k_potsdam.py'
    # checkpoint_file ='/home/ubuntu/paperCode/codeLib/mmsegmentation/work_dirs/fcn_hr18_4x4_512x512_40k_potsdam/best_mIoU_iter_39168_1.pth'
    checkpoint_file = '/home/ubuntu/paperCode/codeLib/mmsegmentation/work_dirs/fcn_hr18_4x4_512x512_40k_potsdam/iter_300.pth'
    model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
    result = inference_segmentor(model, imagePath)

    # show_result_pyplot(model, imagePath, result, get_palette('potsdam'))

    img = model.show_result(
    imagePath, result, palette=get_palette('potsdam'), show=False, opacity=1)
    plt.figure(figsize=(10,10))
    plt.imshow(mmcv.bgr2rgb(img))
    # plt.title(title)
    plt.tight_layout()
    # plt.show(block=block)

# inference_image()
# %%
# show different levels of fog in potsdam
def showFogLevels():
    fig,axs = get_grid(5,1,5,figsize = (50, 10),return_fig=True)
    for i in range(5):
        tempPath = f'/home/ubuntu/paperCode/codeLib/mmsegmentation/swpTest/tempDataTest/potsdam/img_dir/testFog/2_13_1536_4096_2048_4608_fog_{i+1}.png'
        tempImage = mmcv.imread(tempPath)
        [i.set_axis_off() for i in axs]
        axs[i].imshow(tempImage)
    fig.tight_layout()
    plt.savefig('/home/ubuntu/paperCode/codeLib/mmsegmentation/swpTest/swpTestImage/fogLevels.png',dpi=400)
# showFogLevels()
def showFogLevels2():
    fig,axs = get_grid(5,1,5,figsize = (50, 10),return_fig=True)
    # original = mmcv.imread('/home/ubuntu/paperCode/codeLib/mmsegmentation/swpTest/tempDataTest/vaihingen/img_dir/testFog/area2_0_2048_512_2560.png')
    
    for i in range(5):
        tempPath = f'/home/ubuntu/paperCode/codeLib/mmsegmentation/swpTest/tempDataTest/vaihingen/img_dir/testFog/area2_0_2048_512_2560_fog_{i+1}.png'
        tempImage = mmcv.imread(tempPath)
        [i.set_axis_off() for i in axs]
        axs[i].imshow(tempImage)
    fig.tight_layout()
    plt.savefig('/home/ubuntu/paperCode/codeLib/mmsegmentation/swpTest/swpTestImage/fogLevels3.png',dpi=400)
# showFogLevels2()
#%%
# img与dsm进行融合
import cv2
img_dir = '/home/swp/paperCode/IGRLCode/mmf/optical/vaihingen/img_dir/val'
dsm_dir = '/home/swp/paperCode/IGRLCode/mmf/optical/vaihingen/dsm_dir/val'


def convert2RGBA(img_dir,dsm_dir):
    imgs = get_image_files(img_dir)
    imgs.sort()
    dsms = get_image_files(dsm_dir)
    dsms.sort()
    for i in range(len(imgs)):
        pathName, fileName = os.path.split(imgs[i])
        prefix_name = os.path.splitext(fileName)[0]
        suffix_name = os.path.splitext(fileName)[1]
                # read the rgb fog file, 原先的rgb图像，其实保存的时候是bgr格式
        img = mmcv.imread(imgs[i],flag='unchanged')
                # convert it to rgba format
        rgba = cv2.cvtColor(img,cv2.COLOR_RGB2BGRA)
        # extract the alpha channel from file in the val dataset
        dsm = mmcv.imread(dsms[i],flag='unchanged')
        rgba[:,:,3] = dsm
        mmcv.imwrite(rgba,'/home/swp/paperCode/IGRLCode/mmf/optical/vaihingen/img_dsm_dir/val/'+fileName)

# convert2RGBA(img_dir,dsm_dir)       
#%%
def convertBGR2RGB(img_dir):
    imgs = get_image_files(img_dir)
    for i in range(len(imgs)):
        pathName, fileName = os.path.split(imgs[i])
        prefix_name = os.path.splitext(fileName)[0]
        suffix_name = os.path.splitext(fileName)[1]
                # read the rgb fog file, 原先的rgb图像，其实保存的时候是bgr格式
        img = mmcv.imread(imgs[i],channel_order='rgb')
                # convert it to rgba format
        # rgba = cv2.cvtColor(img,cv2.COLOR_RGB2BGRA)
        mmcv.imwrite(img, pathName+'/convert/'+fileName)

# convertBGR2RGB('/home/swp/dataFiles/zhangzhe/rsData/optical/vaihingen/img_dir')     

# %%
