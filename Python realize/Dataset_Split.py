# coding=utf-8
import math
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import torch.utils.data as Data
from models.Student_Class import StudentNet2 as StudentModel
import pandas as pd
#初始参数部分，在这里改动


import argparse
parser = argparse.ArgumentParser(description='Hyperparameter')
parser.add_argument('--input_image_size', type=int, default=96, help='The input_image_size')
parser.add_argument('--LR', type=int, default=0.0005, help='The learning rate')
parser.add_argument('--BatchSize', type=int, default=8, help='The BatchSize')
parser.add_argument('--Epoch', type=int, default=30, help='The epoch')
parser.add_argument('--Temperature', type=float, default=[1.0, 2.0, 5.0, 7.0, 10.0, 20.0, 50.0], help='The Temperature')
parser.add_argument('--Source_data_path', type=str, default='./data/p2/', help='The Source data path')
parser.add_argument('--Dataset_save_path', type=str, default='./Dataset/', help='The Dataset_save_path')

parser.add_argument('--Model_save_path', type=str, default='./models/', help='The Trained model save path')
parser.add_argument('--Result_save_path', type=str, default='./results/', help='The result save path')
args = parser.parse_args()

#加载数据集
data_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5)),

    ])
data_set = ImageFolder(args.Source_data_path, transform=data_transform)
n = len(data_set)           #示例总数
indice = [x for x in range(n)]
# random.seed(30)
random.seed(100)
random.shuffle(indice)
indice_test = indice[:n*2 // 5]     #40%做测试集
indice_train = indice[n*2 // 5:]
# indice_test = indice[:n*5 // 10]     #50%做测试集
# indice_train = indice[n*5 // 10:]
print('训练集大小：{}'.format(len(indice_train)))
print('测试集大小：{}'.format(len(indice_test)))
#n_test = int(0.1 * n)          #花费〜10％进行测试,是不是该产生随机索引进行取数据集
#test_set = Data.Subset(data_set, range(n_test))  # take first 10%

#train_set = Data.Subset(data_set, range(n_test, n))  # take the rest

test_set = Data.Subset(data_set, indice_test)  # take first 40%
train_set = Data.Subset(data_set, indice_train)  # take first 60%
cou_dic = {}
print(len(test_set.dataset.imgs))
print(len(test_set))
# print(len(test_set.dataset.imgs[indice_test]))
for idx in indice_test:
# for item in test_set.dataset.imgs[indice_test]:
    if test_set.dataset.imgs[idx][1] in cou_dic.keys():
        cou_dic[test_set.dataset.imgs[idx][1]] += 1
    else:
        cou_dic[test_set.dataset.imgs[idx][1]] = 1
print(cou_dic)
'''
https://blog.csdn.net/qq_42255269/article/details/109243297
'''
# torch.save(train_set, args.Dataset_save_path + '/train_set_p2_randomChange50.pt')
# torch.save(test_set, args.Dataset_save_path + '/test_set_s1_p2randomChange50.pt')
