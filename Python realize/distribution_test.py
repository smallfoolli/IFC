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

#torch.manual_seed(0)
#torch.cuda.manual_seed(0)

#关键，定义kd的loss

def distillation(y, labels, teacher_scores, temp, alpha):
    return nn.KLDivLoss()(F.log_softmax(y / temp, dim=1), F.softmax(teacher_scores / temp, dim=1)) * (
            temp * temp * 2.0 * alpha) + F.cross_entropy(y, labels) * (1. - alpha)


def KD_train_student(model, teacher_model, device, train_loader, optimizer, epoch, temperature, alpha):
    model.train()
    teacher_model.eval()
    trained_samples = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        teacher_output = teacher_model(data)
        teacher_output = teacher_output.detach()
        loss = distillation(output, target, teacher_output, temp=temperature, alpha=alpha)
        #loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        trained_samples += len(data)
        progress = math.ceil(batch_idx / len(train_loader) * 50)
        print("\rTrain epoch %d: %d/%d" %
              (epoch, trained_samples, len(train_loader.dataset)), end='')


def KD_test_student(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest: average loss: {:.4f}, accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss, correct / len(test_loader.dataset)


def KD_student_train_main(temperature, alpha, args):
    #加载模型，并进行训练
    torch.manual_seed(0)
    teacher_model = torch.load(args.Model_save_path + '/Trained teacher model on datasetp2')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model = teacher_model.to(device)
    model = StudentModel(args.input_image_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.LR, weight_decay=1e-5)
    student_history = []
    for epoch in range(1, args.Epoch + 1):
        KD_train_student(model, teacher_model, device, train_loader, optimizer, epoch, temperature, alpha)
        loss, acc = KD_test_student(model, device, test_loader)
        student_history.append((epoch, temperature, alpha, loss, acc))
    #保存下model
    torch.save(model, args.Model_save_path + 'Student2T' + str(temperature) + 'Alpha' + str(alpha))
    return student_history
#保存整个模型
#torch.save(model, './modelstore/model_before_prue')
#model = torch.load('./modelstore/letter_class_feature932515')

# 训练学生网络
train_history = []
for temperature in args.Temperature:
    for alpha in args.Alpha:
        #传参输出数据
        train_history_tem = KD_student_train_main(temperature, alpha, args)
        train_history.extend(train_history_tem)
        #torch.cat([train_history, train_history_tem], dim=0)




#student_model, KD_student_history = KD_student_main(args, train_loader, test_loader)
#可以存储一下结果，存成CSV或者Excel
KD_student_history = train_history
result_total = {'ID': [i+1 for i in range(len(KD_student_history))],
                'Epoch': [KD_student_history[i][0] for i in range(len(KD_student_history))],
                'KD temperature': [KD_student_history[i][1] for i in range(len(KD_student_history))],
                'KD alpha': [KD_student_history[i][2] for i in range(len(KD_student_history))],
                'Test Loss': [KD_student_history[i][3] for i in range(len(KD_student_history))],
                'Test Accuracy': [KD_student_history[i][4] for i in range(len(KD_student_history))]}
df_result = pd.DataFrame(result_total)
print(df_result)
#df_result.to_csv(args.Result_save_path + 'Student2 with KD results on datasetp20007A09TVEMaxJix_random.csv', index=False, sep=',')
