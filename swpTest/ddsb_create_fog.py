# -*- coding: utf-8 -*-
# %%
import os.path as osp
import mmcv
import matplotlib.colors as colors
import collections
import os
from PIL import Image
import os.path
import time
import torch
import torchvision.datasets as dset
import torchvision.transforms as trn
import torch.utils.data as data
import numpy as np
from fastai.basics import *
from fastai.vision import *
from PIL import Image
import skimage as sk
from skimage.filters import gaussian
from io import BytesIO
# from wand.image import Image as WandImage
# from wand.api import library as wandlibrary
# import wand.color as WandColor
import ctypes
from PIL import Image as PILImage
import cv2
from scipy.ndimage import zoom as scizoom
from scipy.ndimage.interpolation import map_coordinates
import warnings
# %%
warnings.simplefilter("ignore", UserWarning)
# /////////////// Data Loader ///////////////

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png',
                  '.ppm', '.bmp', '.pgm', '.tiff', '.tif']


def is_image_file(filename):
    """Checks if a file is an image.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(
        dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        # img = img.resize((256, 256),Image.ANTIALIAS)
        imgNp = np.asarray(img)
        alphaData = None
        if imgNp.shape[-1] == 4:
            # denote the RGBA files,save the alpha channel
            alphaData = imgNp[:, :, -1]
        imgRGB = img.convert('RGB')
        return imgRGB, alphaData

# %%


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def plasma_fractal(mapsize=512, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
                 stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize,
                          stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize //
                 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize,
                 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def fog(x, severity=1):
    c = [(1.5, 2), (2, 2), (2.5, 1.7), (2.5, 1.5), (3, 1.4)][severity - 1]
    if type(x) != ndarray:
        x = np.array(x) / 255.
    else:
        x = x/255
    max_val = x.max()
    x += c[0] * plasma_fractal(wibbledecay=c[1])[:300, :300][..., np.newaxis]
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255


print('\nUsing ImageNet data')

d = collections.OrderedDict()
d['Fog'] = fog
# %%
# read an image
temp_path = Path('tempDataTest/img_dir/test/')
fileLists = get_image_files(temp_path)
# %%
# make a dataset using pytorch from the folder teseImages and have a transform than save it
# to a new folder


class fogAugmented(data.Dataset):
    def __init__(self, root, method, severity, transform=None, target_transform=None,
                 loader=default_loader):
        # root: e.g. '/home/ubuntu/paperCode/codeLib/mmsegmentation/swpTest/tempDataTest/vaihingen/img_dsm_dir/'
        fileLists = get_image_files(Path(root+'val/'))
        if len(fileLists) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                                                                                 IMG_EXTENSIONS)))
        self.root = root
        self.method = method
        self.severity = severity
        self.filePath = fileLists
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        alpha = None
        path = self.filePath[index]
        ori = mmcv.imread(path, flag='unchanged')
        if ori.shape[-1] == 4:
            img = ori[:, :, :3]
            alpha = ori[:, :, -1]
        else:
            img = ori
        # img,alphaData = self.loader(path)
        pathName, fileName = os.path.split(path)
        prefix_name = os.path.splitext(fileName)[0]
        suffix_name = os.path.splitext(fileName)[1]
        img = self.method(img, self.severity)
        # if self.transform is not None:
        #     img = self.transform(img)
        #     img = self.method(img, self.severity)
        replacedName = self.root + 'testFog/' + prefix_name + \
            '_fog_'+str(self.severity)+suffix_name
        print(replacedName)

        # replacedName += path[path.rindex('/'):]
        # convert to RGBA data format
        if alpha is not None:
            ori[:, :, :3] = img
            ori[:, :, -1] = alpha
        else:
            ori = img
            # save the image using the unchanged file format
        mmcv.imwrite(ori, replacedName)

        # Image.fromarray(np.uint8(img)).save(
        #     replacedName, quality=99, optimize=True)

        return 0  # we do not care about returning the data

    def __len__(self):
        return len(self.filePath)
# %%

def save_fog(path, method=fog):
    for severity in range(1, 6):
        print(method.__name__, severity)
        distorted_dataset = fogAugmented(
            # root=path+'/img_dsm_dir/',
            root=path+'/img_dir/',
            method=method, severity=severity, transform=trn.Compose(
                [trn.RandomCrop(300)])
        )
        distorted_dataset_loader = torch.utils.data.DataLoader(
            distorted_dataset, batch_size=5, shuffle=False, num_workers=0)

        for _ in distorted_dataset_loader:
            continue


# %%
# save_fog('/home/swp/paperCode/IGRLCode/mmf/optical/vaihingen/',fog)

# %%
# create the corresponding label in ann_dir
def copyAndRename(path):
    # copy the files in ann_dir/testFog five times and rename
    fileLists = get_image_files(Path(path+'val/'))
    mmcv.mkdir_or_exist(path+'testFog/')
    for file in fileLists:
        pathName, fileName = os.path.split(file)
        prefix_name = os.path.splitext(fileName)[0]
        suffix_name = os.path.splitext(fileName)[1]
        for i in range(5):
            newName = path+'testFog/'+prefix_name+'_fog_'+str(i+1)+suffix_name
            print(newName)
            shutil.copy(file, newName)
        # os.remove(file)

#%%
def makeFogCorruptedDataset(path):
    # copy  img_dir/val to img_dir/testFog
    # path = '/home/ubuntu/paperCode/codeLib/mmsegmentation/swpTest/tempDataTest/vaihingen/'
    mmcv.mkdir_or_exist(path+'/img_dir/testFog/')
    mmcv.mkdir_or_exist(path+'/ann_dir/testFog/')
    save_fog(path)
    # annotations copy
    copyAndRename(path+'/ann_dir/')


# %%
makeFogCorruptedDataset('/home/swp/paperCode/IGRLCode/mmf/tempDataset/ddsb2')

# %%
# generate the testFog in dsm_dir
# copyAndRename('/home/swp/paperCode/IGRLCode/mmf/tempDataset/potsdam/dsm_dir/')
#%%
# using img_dir and dsm_dir to create img_dsm_dir

def convert2RGBA(img_dir,dsm_dir,img_dsm_dir):
    mmcv.mkdir_or_exist(osp.join(img_dsm_dir,  'train'))
    mmcv.mkdir_or_exist(osp.join(img_dsm_dir, 'val'))
    mmcv.mkdir_or_exist(osp.join(img_dsm_dir,  'testFog'))
    for temp in ['train/','val/','testFog/']:
        img_dir_temp = img_dir+temp
        dsm_dir_temp = dsm_dir+temp

        imgs = get_image_files(img_dir_temp)
        imgs.sort()
        dsms = get_image_files(dsm_dir_temp)
        dsms.sort()
        for i in range(len(imgs)):
            pathName, fileName = os.path.split(imgs[i])
            prefix_name = os.path.splitext(fileName)[0]
            suffix_name = os.path.splitext(fileName)[1]
                    # read the rgb fog file, 原先的rgb图像，其实保存的时候是bgr格式
            img = mmcv.imread(imgs[i],flag='unchanged')
                    # convert it to rgba format
            # rgba = cv2.cvtColor(img,cv2.COLOR_RGB2BGRA)
            rgba = cv2.cvtColor(img,cv2.COLOR_BGR2BGRA)
            # extract the alpha channel from file in the val dataset
            dsm = mmcv.imread(dsms[i],flag='unchanged')
            rgba[:,:,3] = dsm
            mmcv.imwrite(rgba, img_dsm_dir + temp+fileName)

img_dir = '/home/swp/paperCode/IGRLCode/mmf/tempDataset/potsdam/img_dir/'
dsm_dir = '/home/swp/paperCode/IGRLCode/mmf/tempDataset/potsdam/dsm_dir/'
img_dsm_dir = '/home/swp/paperCode/IGRLCode/mmf/tempDataset/potsdam/img_dsm_dir/'
# convert2RGBA(img_dir,dsm_dir,img_dsm_dir)       
#%%
