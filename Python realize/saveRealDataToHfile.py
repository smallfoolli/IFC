

import numpy as np
import os

import torch.nn as nn

import pandas as pd
import time
import torch
import matplotlib.pyplot as plt
import confusion as cf
import argparse
import torch.utils.data as Data
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder

parser = argparse.ArgumentParser(description='Hyperparameter')
parser.add_argument('--input_image_size', type=int, default=96, help='The input_image_size')
parser.add_argument('--LR', type=int, default=0.0003, help='The learning rate')
parser.add_argument('--BatchSize', type=int, default=1, help='The BatchSize')
parser.add_argument('--Epoch', type=int, default=50, help='The epoch')
parser.add_argument('--Dataset_save_path', type=str, default='./Dataset/', help='The Dataset_save_path')
parser.add_argument('--Model_save_path', type=str, default='./models/', help='The Trained model save path')
# parser.add_argument('--Source_data_path2', type=str, default='./data/24Test/', help='The Source data path')
# parser.add_argument('--Source_data_path1', type=str, default='./data/24Train/', help='The Source data path')

# parser.add_argument('--Source_data_path1', type=str, default='./radarDataNew/data2_12/1', help='The Source data path')
# parser.add_argument('--Source_data_path2', type=str, default='./radarDataNew/data2_12/2', help='The Source data path')

parser.add_argument('--Source_data_path1', type=str, default='./radarDataNew/data3_12/train', help='The Source data path')
parser.add_argument('--Source_data_path2', type=str, default='./radarDataNew/data3_12/test', help='The Source data path')


args = parser.parse_args()
def randomSeed():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def Samples2H(sample, file):
    # w = sample.data.size(0)
    c = sample.data.size(0)
    y = sample.data.size(1)
    x = sample.data.size(2)
    file.write('{\n')
    for j in range(c):
        file.write('{')
        for k in range(y):
            file.write('{')
            for m1 in range(x):
                file.write(str(sample.data[j][k][m1].item()))
                if m1 != x - 1:
                    file.write(',')
            file.write('}')
            if k != y - 1:
                file.write(',')
        file.write('}')
        if j != c - 1:
            file.write(',')
        file.write('\n')
    file.write('}')






if __name__ == '__main__':
    # 加载数据集
    data_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5)),

        ])
    train_set = ImageFolder(args.Source_data_path1, transform=data_transform)
    test_set = ImageFolder(args.Source_data_path2, transform=data_transform)
    # randomSeed()
    data_loader = Data.DataLoader(dataset=test_set, batch_size=args.BatchSize, shuffle=True)
    for inputs, targets in data_loader:
        print(targets)
        x = inputs.squeeze()
        # y = x.permute(1, 2, 0)
        # pad = nn.ZeroPad2d(padding=(1, 1, 1, 1))
        # y = pad(x)
        with open('sample4.h', 'w') as file:
            Samples2H(x, file)

        print('完成！')
        break


