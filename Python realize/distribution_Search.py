#utf-8
import math
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import torch.utils.data as Data
from models.Student_Class import StudentNet2 as StudentModel
import matplotlib.pyplot as plt
#从pyplot导入MultipleLocator类，这个类用于设置刻度间隔
import pandas as pd
#初始参数部分，在这里改动


import argparse
parser = argparse.ArgumentParser(description='Hyperparameter')
parser.add_argument('--input_image_size', type=int, default=96, help='The input_image_size')
parser.add_argument('--LR', type=int, default=0.0007, help='The learning rate')
parser.add_argument('--BatchSize', type=int, default=8, help='The BatchSize')
parser.add_argument('--Epoch', type=int, default=50, help='The epoch')
#parser.add_argument('--Temperature', type=float, default=[1.0, 2.0, 5.0, 7.0, 10.0, 20.0, 50.0, 75.0, 100.0, 125.0,150.0, 200.0, 250.0, 500.0, 500000.0], help='The Temperature')
parser.add_argument('--Temperature', type=float, default=[1.0], help='The Temperature')
parser.add_argument('--Alpha', type=float, default=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0], help='The Alpha')
#parser.add_argument('--Temperature', type=float, default=[1.0, 2.0, 5.0, 7.0, 10.0, 20.0, 50.0], help='The Temperature')
#parser.add_argument('--Alpha', type=float, default=[0.9], help='The Alpha')
parser.add_argument('--Source_data_path', type=str, default='./data/s1_1.5m/', help='The Source data path')
parser.add_argument('--Dataset_save_path', type=str, default='./Dataset/', help='The Dataset_save_path')
#parser.add_argument('--Alpha', type=float, default=[0.9, 0.8, 0.7, 0.6, 0.5], help='The Alpha')
'''
../ 表示当前文件所在的目录的上一级目录
./ 表示当前文件所在的目录(可以省略)
/ 表示当前站点的根目录(域名映射的硬盘目录)
'''
parser.add_argument('--Model_save_path', type=str, default='./models/', help='The Trained model save path')
parser.add_argument('--Result_save_path', type=str, default='./results/', help='The result save path')
args = parser.parse_args()

#加载数据集
train_set = torch.load(args.Dataset_save_path + '/train_set_p2.pt')
test_set = torch.load(args.Dataset_save_path + '/test_set_s1_p2.pt')

train_loader = Data.DataLoader(dataset=train_set, batch_size=args.BatchSize, shuffle=True)
test_loader = Data.DataLoader(dataset=test_set, batch_size=args.BatchSize, shuffle=True)

#加载不同的网络模型，对每个网络模型进行测试，存储其输出分布信息，最后取数画图？
#是否需要每个网络都取每个类别并画图？图像数太多

def KD_test_student(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    pro_vector_val = torch.zeros(1, 10)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pro_vector = F.softmax(output, dim=1)
            pro_vector_val = torch.cat([pro_vector_val, pro_vector.cpu()])
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest: average loss: {:.4f}, accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss, correct / len(test_loader.dataset), pro_vector_val
x = [i for i in range(10)]
#data_Pro_all = torch.zeros(1, 10)
data_All = {}
for temperature in args.Temperature:
    for alpha in args.Alpha:
        model = torch.load(args.Model_save_path + 'Student2T' + str(temperature) + 'Alpha' + str(alpha))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        v1, v2, data_Pro = KD_test_student(model, device, test_loader)
        #data_Pro_all = torch.cat([data_Pro_all, data_Pro])
        data_All[str(temperature)+str(alpha)] = data_Pro
        for j in range(1,len(data_Pro.numpy()[:,1])):
            plt.figure()
            plt.stem(x, data_Pro.numpy()[j, :])
            plt.ylabel('Probability')
            plt.xlabel('Class index')  # 我们设置横纵坐标的标题。
            plt.title('Probability distribution')
            plt.xticks([i for i in range(10)])
            #plt.xlim(-0.5, 9.5)
            plt.show()
            plt.close()

#取数画图
print(data_All)
#res = pd.DataFrame(data_All, index=[i for i in range(84)])
#res = pd.DataFrame(data_All, index=[0])
#res.to_csv(args.Result_save_path + 'Student2 with KD results on datasetp20007AVT1_random_distribution.csv', index=False, sep=',')