#%%
from torchvision.datasets import ImageFolder
import torch
from torchvision import transforms

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
train_dataset = ImageFolder(root='/home/swp/paperCode/IGRLCode/mmf/optical/vaihingen/img_dir/train/', transform=None)
print(getStat(train_dataset))

# %%
