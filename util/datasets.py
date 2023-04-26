import os
import torch
import torchvision
import torchvision.datasets as datasets

root = '..\\data\\minist'
# TODO
def build_dataset(is_train, args):
    if is_train == True:
        path = os.path.join(root, 'train')
    else:
        path = os.path.join(root, 'val')

    return datasets.MNIST(
            root = path,  #数据集的位置
            train = is_train,       #如果为True则为训练集，如果为False则为测试集
            transform = torchvision.transforms.ToTensor(),   #将图片转化成取值[0,1]的Tensor用于网络处理
            download=True)
