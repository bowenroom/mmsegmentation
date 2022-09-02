#%%
import imp
from torchvision.datasets import ImageFolder
import torch
from torchvision import transforms
from fastai.vision import *
def getStat(train_data):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(4)
    std = torch.zeros(4)
    for X, _ in train_loader:
        for d in range(4):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
#%%
# train_dataset = ImageFolder(root='/home/swp/paperCode/IGRLCode/mmf/optical/vaihingen/img_dir/train/', transform=None)
# print(getStat(train_dataset))

# %%
# data augmentation
import albumentations as A
import cv2
from matplotlib import pyplot as plt
# %%
transform = A.Compose([
    A.RandomCrop(width=512, height=512),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.7),
    A.SafeRotate(),
    A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.2, always_apply=False, p=0.6)
])
import matplotlib.colors as colors
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
img = cv2.imread('/home/swp/paperCode/IGRLCode/mmf/optical/vaihingen/img_dir/train/area1_0_0_512_512.png')
mask = cv2.imread('/home/swp/paperCode/IGRLCode/mmf/optical/vaihingen/ann_dir/train/area1_0_0_512_512.png',flags=-1)
dsm = cv2.imread('/home/swp/paperCode/IGRLCode/mmf/optical/vaihingen/dsm_dir/train/area1_0_0_512_512.tiff',flags=-1)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
masks = [mask,dsm]
transformed = transform(image = img, masks = masks)
transformed_image = transformed['image']
transformed_mask = transformed['masks']
fig, axs = plt.subplots(nrows=1,ncols=4, figsize=(10, 40))
axs[0].imshow(img)
axs[1].imshow(transformed_image)
axs[2].imshow(transformed_mask[0],cmap = my_cmap)
axs[3].imshow(transformed_mask[1],cmap = 'Greys')
for ax in axs:
        ax.set_axis_off()
plt.tight_layout()
plt.savefig('/home/swp/paperCode/IGRLCode/mmf/swpTest/transformed2.png',dpi=800)

# %%

# %%

# %%
