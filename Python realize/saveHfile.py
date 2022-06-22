'''
1.
CNN参数输出脚本，自动转换为数组形式的.h代码
2.
支持卷积层和全连接层的组合网络结构
3.
卷积层的权重经过重新组织以便于计算：
[w][c][y][x]
x - 卷积核x坐标
y - 卷积核y坐标
c - 输入矩阵通道数
w - 输出通道数
'''
import pandas as pd
import time
import torch

import argparse

from models.Student_net_versatile_feature_map_Combination import VConv2d

import torch.nn as nn
parser = argparse.ArgumentParser(description='Hyperparameter')
parser.add_argument('--input_image_size', type=int, default=96, help='The input_image_size')
parser.add_argument('--LR', type=int, default=0.0003, help='The learning rate')
parser.add_argument('--BatchSize', type=int, default=8, help='The BatchSize')
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

def Weights2H(model, file):
    for m in model.modules():
        print(m)
        if isinstance(m, VConv2d):
            w = m.weight.data.size(0)
            c = m.weight.data.size(1)
            x = m.weight.data.size(2)
            y = m.weight.data.size(3)
            file.write('{\n')
            for i in range(w):
                file.write('{\n')
                for j in range(c):
                    file.write('{')
                    for k in range(y):
                        file.write('{')
                        for m1 in range(x):
                            file.write(str(m.weight.data[i][j][k][m1].item()))
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
                if i != w - 1:
                    file.write(',')
                file.write('\n')
            file.write('},\n')

            if m.bias is not None:
                w = m.bias.data.size(0)
                file.write('{\n')
                for i in range(w):
                    file.write(str(m.bias.data[i].item()))
                    if i != w - 1:
                        file.write(',')
                file.write('\n},\n')

            if m.weightFeatureMap is not None:
                w = m.weightFeatureMap.data.size(0)
                file.write('{\n')
                for i in range(w):
                    file.write(str(m.weightFeatureMap.data[i].item()))
                    if i != w - 1:
                        file.write(',')
                file.write('\n},\n')

        elif isinstance(m, nn.BatchNorm2d):
            # https://blog.csdn.net/qq_39777550/article/details/108038677，BN层作用，解释的很好
            w = m.weight.data.size(0)
            file.write('{\n')
            for i in range(w):
                file.write(str(m.weight.data[i].item()))
                if i != w - 1:
                    file.write(',')
            file.write('\n},\n')
            # m.weight.data.fill_(0.5)
            # m.bias.data.zero_()
            w = m.bias.data.size(0)
            file.write('{\n')
            for i in range(w):
                file.write(str(m.bias.data[i].item()))
                if i != w - 1:
                    file.write(',')
            file.write('\n},\n')
        elif isinstance(m, nn.Linear):
            print(m.weight.data.size())
            print(m.bias.data.size())
            w = m.weight.data.size(0)
            c = m.weight.data.size(1)

            file.write('{\n')
            for i in range(w):
                file.write('{')
                for j in range(c):
                    file.write(str(m.weight.data[i][j].item()))
                    if j != c - 1:
                        file.write(',')
                file.write('}')
                if i != w - 1:
                    file.write(',')
                file.write('\n')
            file.write('},\n')

            w = m.bias.data.size(0)
            file.write('{\n')
            for i in range(w):
                file.write(str(m.bias.data[i].item()))
                if i != w - 1:
                    file.write(',')
            file.write('\n},\n')
            # m.weight.data.normal_(0, 0.01)
            # m.bias.data.zero_()


if __name__ == '__main__':
    # file = open('weights-CNN.h', 'w')
    with open('CustomCNN_weights_net9.h', 'w') as file:
        teacher_model = torch.load(args.Model_save_path + '/Trained KDstudent9_versatile model on data3 cross person')
        # weights = model.get_weights()
        # weights = teacher_model.parameters()
        Weights2H(teacher_model, file)
        print('hh')
    # file.close()