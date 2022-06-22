#!/usr/bin/env python
# coding=utf-8
import math
import torch
import time
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import torch.utils.data as Data
# from models import Student_net_versatile_for_performance as models
# from models import Student_net_versatile_feature_map_Combination as models
from models import Student_net_versatile_feature_map_Combination_Norm as models
#from models.Teacher_Class import TeacherNet as TeacherModel
import pandas as pd

# torch.optim.lr_scheduler.ExponentialLR()
#初始参数部分，在这里改动
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

import argparse
parser = argparse.ArgumentParser(description='Hyperparameter')
parser.add_argument('--input_image_size', type=int, default=96, help='The input_image_size')
# parser.add_argument('--LR', type=int, default=0.00001, help='The learning rate')
parser.add_argument('--LR', type=int, default=0, help='The learning rate')
parser.add_argument('--LRW', type=int, default=0.08, help='The learning rate')
parser.add_argument('--BatchSize', type=int, default=8, help='The BatchSize')
parser.add_argument('--Epoch', type=int, default=55, help='The epoch')
parser.add_argument('--Temperature', type=float, default=[1.0, 2.0, 5.0, 7.0, 10.0, 20.0, 50.0], help='The Temperature')
parser.add_argument('--Source_data_path', type=str, default='./data/s1_1.5m/', help='The Source data path')
parser.add_argument('--Dataset_save_path', type=str, default='./Dataset/', help='The Dataset_save_path')
parser.add_argument('--pretrained', default=True, dest='pretrained', action='store_true', help='use pre-trained model')
'''
../ 表示当前文件所在的目录的上一级目录
./ 表示当前文件所在的目录(可以省略)
/ 表示当前站点的根目录(域名映射的硬盘目录)
'''
parser.add_argument('--Model_save_path', type=str, default='./models/', help='The Trained model save path')
parser.add_argument('--Result_save_path', type=str, default='./result/', help='The result save path')

# parser.add_argument('--Source_data_path2', type=str, default='./data/24Test/', help='The Source data path')
# parser.add_argument('--Source_data_path1', type=str, default='./data/24Train/', help='The Source data path')

# parser.add_argument('--Source_data_path1', type=str, default='./radarDataNew/data2_12/1', help='The Source data path')
# parser.add_argument('--Source_data_path2', type=str, default='./radarDataNew/data2_12/2', help='The Source data path')

parser.add_argument('--Source_data_path1', type=str, default='./radarDataNew/data3_12/train', help='The Source data path')
parser.add_argument('--Source_data_path2', type=str, default='./radarDataNew/data3_12/test', help='The Source data path')

parser.add_argument('--arch', '-a', metavar='ARCH', default='Student', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: Student)')
args = parser.parse_args()

#加载数据集
def randomSeed():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
# torch.manual_seed(0)
# torch.cuda.manual_seed(0)
randomSeed()
# train_set = torch.load(args.Dataset_save_path + '/train_set_p2.pt')
# test_set = torch.load(args.Dataset_save_path + '/test_set_s1_p2.pt')

# train_set = torch.load(args.Dataset_save_path + '/train_set_s1_1.5m.pt')
# test_set = torch.load(args.Dataset_save_path + '/test_set_s1_1.5m.pt')
data_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5)),

    ])
train_set = ImageFolder(args.Source_data_path1, transform=data_transform)
test_set = ImageFolder(args.Source_data_path2, transform=data_transform)


train_loader = Data.DataLoader(dataset=train_set, batch_size=args.BatchSize, shuffle=True)
test_loader = Data.DataLoader(dataset=test_set, batch_size=args.BatchSize, shuffle=True)



def train_teacher(model, device, train_loader, optimizer, epoch):
    randomSeed()
    model.train()
    trained_samples = 0
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        trained_samples += len(data)
        progress = math.ceil(batch_idx / len(train_loader) * 50)
        print("\rTrain epoch %d: %d/%d" %
              (epoch, trained_samples, len(train_loader.dataset)), end='')
    train_loss /= len(train_loader.dataset)
    print(' Train loss is: {:.4f}'.format(train_loss))
    return train_loss


def test_teacher(model, device, test_loader):
    randomSeed()
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

    print('Test: average loss: {:.4f}, accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss, correct / len(test_loader.dataset)


def teacher_main(args, train_loader, test_loader):
    # torch.manual_seed(0)
    # torch.cuda.manual_seed(0)
    randomSeed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()
    #print("=> creating model '{}'".format(args.arch))
    #model = models.__dict__[args.arch]()
    #查看一下参数类型有哪些
    # for name, para in model.named_parameters():
    #     print(name)
    model = model.to(device)
    # tem = model.parameters()
    #对参数进行分组，然后分组设置学习率，参考https://blog.csdn.net/junqing_wu/article/details/94395340
    para_w = []
    para_others = []
    for name, para in model.named_parameters():
        if 'weightFeatureMap' in name:
            para_w += [para]
        else:
            para_others += [para]
        # print(name)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.LR, weight_decay=1e-5)
    # optimizer = torch.optim.Adam([{'params': para_w, 'lr': args.LRW},
    #                               {'params': para_others, 'lr': args.LR / (5**5)}], weight_decay=1e-5)

    optimizer = torch.optim.Adam([{'params': para_w, 'lr': args.LRW},
                                  {'params': para_others, 'lr': args.LR}], weight_decay=1e-5)

    teacher_history = []
    LR_W = args.LRW
    LR = args.LR

    for epoch in range(1, args.Epoch + 1):
        # LR change，跨场景是每隔5轮换一次
        # if epoch > 0:
        if epoch % 5 == 0:
            LR_W = LR_W / 5
            LR = LR / 5
        #     LR_W = args.LRW*(0.98**(epoch - 1))
        #     LR = args.LR * (0.8 ** (epoch - 1))
            optimizer = torch.optim.Adam([{'params': para_w, 'lr': LR_W},
                                  {'params': para_others, 'lr': LR}], weight_decay=1e-5)
        train_L = train_teacher(model, device, train_loader, optimizer, epoch)
        # loss, acc = test_teacher(model, device, test_loader)
        # # 打印权值看有没有变
        # # for name, para in model.named_parameters():
        # #     if 'weightFeatureMap' in name:
        # #         print(name, para)
        # teacher_history.append((loss, acc, train_L))

    #torch.save(model.state_dict(), "teacher.pt")
    #torch.save(model, args.Model_save_path + '/Trained student_versatile model on datasets115 and trained model')
    #torch.save(model, args.Model_save_path + '/Trained student_versatile_delta2 model on datasetp2 and trained model')
    #torch.save(model, args.Model_save_path + '/Trained student_versatile_delta0_g2 model on datasetp2 and trained model')
    #torch.save(model, args.Model_save_path + '/Trained student6_versatile model on datasets115 and notrained model')
    # torch.save(model, args.Model_save_path + '/Trained KDstudent9_versatile model on data2 cross scenario')
    # torch.save(model, args.Model_save_path + '/Trained KDstudent9_versatile model on data3 cross person')
    # torch.save(model, args.Model_save_path + '/Trained KDstudent9_versatile model on data3 cross person LR LRW change')
    # torch.save(model, args.Model_save_path + '/Trained student9_versatile model withoutKD on data2 cross scenario')
    return model, teacher_history

#保存整个模型
#torch.save(model, './modelstore/model_before_prue')
#model = torch.load('./modelstore/letter_class_feature932515')
data_result = {}
# 训练教师网络
# LRW = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4]
# LRW = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1]
# LRW = [5e-4]
LRW = [0.1]
# LRW = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1]
# LRW = [0.1, 0.3, 0.5, 0.7, 0.9]
# LRW = [1e-3, 2.5e-3, 5e-3, 7.5e-3, 1e-2, 5e-2, 0.1, 0.25, 0.5, 0.75]
# LRW = [1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.25]
# LRW = [item / 4000 for item in range(1, 401, 4)]
# LRW = [0.005]
# LRW = [item / 1000 for item in range(1, 501, 4)]
for val in LRW:
    args.LRW = val
    print('The LRW is: {}'.format(val))
    start_time = time.time()
    teacher_model, teacher_history = teacher_main(args, train_loader, test_loader)
    finish_time = time.time()
    total_time = finish_time - start_time
    print('多尺度学习总用时长为：{}'.format(total_time))
#     data_result['LRW= ' + str(args.LRW) + ' Test Loss'] = [teacher_history[i][0] for i in
#                                                                           range(args.Epoch)]
#     data_result['LRW= ' + str(args.LRW) + ' Acc'] = [teacher_history[i][1] for i in
#                                                                          range(args.Epoch)]
#     data_result['LRW= ' + str(args.LRW) + ' Train Loss'] = [teacher_history[i][2] for i in
#                                                      range(args.Epoch)]
# df_result = pd.DataFrame(data_result)
# print(df_result)
#df_result.to_csv(args.Result_save_path + 'Student6 results_LR_Batchsize on dataset_s115.csv', index=False, sep=',')
# df_result.to_csv(args.Result_save_path + 'Student8 results_LRW_search1_randomStateK5 on datasetp2p1.csv', index=False, sep=',')
# df_result.to_csv(args.Result_save_path + 'Student8 results_LRW_search1_randomStateK5 on dataset24.csv', index=False, sep=',')
# df_result.to_csv(args.Result_save_path + 'Student8 results_LRW_VarFinetune_search1_randomStateK5 weightNot randomV3 on data2 cross scenario.csv', index=False, sep=',')

# df_result.to_csv(args.Result_save_path + 'Student8 results_LRW_Var_search1_K5 on data3 cross person.csv', index=False, sep=',')
# df_result.to_csv(args.Result_save_path + 'Student9 results_LRW_LR_Var_search1_K77 on data2 cross scenario V4.csv', index=False, sep=',')
# df_result.to_csv(args.Result_save_path + 'Student9 results_LRW_LR_Var_search1_K77 on data3 cross person v2.csv', index=False, sep=',')
#
# df_result.to_csv(args.Result_save_path + 'Student9 results_LRW01_normalized_K77 on data3 cross person_LR0006BS4 T20 A08.csv', index=False, sep=',')
#
# df_result.to_csv(args.Result_save_path + 'Student9 results_LRW01_normalized_K77 on data2 cross scenario_LR0006BS8 T20 A08.csv', index=False, sep=',')
#
# df_result.to_csv(args.Result_save_path + 'Student9 results_LRW_normalized_Var_search1_K77 on data3 cross person v4.csv', index=False, sep=',')
#
# df_result.to_csv(args.Result_save_path + 'Student9 results_LRW_Var_search1_K77 on data3 cross person v3epoch9.csv', index=False, sep=',')
# df_result.to_csv(args.Result_save_path + 'Student9 withoutKD results_LRW_Var_search1_K77 on data3 cross person.csv', index=False, sep=',')
# df_result.to_csv(args.Result_save_path + 'Student9 withoutKD results_LRW_Var_search1_K77 on data2 cross scenario V1.csv', index=False, sep=',')

# teacher_model, teacher_history = teacher_main(args, train_loader, test_loader)
#可以存储一下结果，存成CSV或者Excel

# result_total = {'Epoch': [i+1 for i in range(args.Epoch)],
#                 'Test Loss': [teacher_history[i][0] for i in range(args.Epoch)],
#                 'Test Accuracy': [teacher_history[i][1] for i in range(args.Epoch)]}
# df_result = pd.DataFrame(result_total)
# print(df_result)
#df_result.to_csv(args.Result_save_path + 'Teacher results on datasetp2.csv', index=False, sep=',')
'''
#画出训练测试曲线
import matplotlib.pyplot as plt
epochs = args.Epoch
x = list(range(1, epochs+1))

plt.subplot(2, 1, 1)
plt.plot(x, [teacher_history[i][1]*100 for i in range(epochs)], label='teacher')
plt.xlim(1, args.Epoch)
#plt.xticks(x)  # 设置x刻度
#plt.yticks([0,0.2,0.4,0.6,0.8,1.0,1.2])  # 设置y刻度
plt.xlabel('Epoch number')
plt.ylabel('Accuracy(%)')
'''
#https://blog.csdn.net/leilei9406/article/details/84103579?utm_medium=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.control&dist_request_id=&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.control
'''
plt.title('Test accuracy')
plt.legend()


plt.subplot(2, 1, 2)
plt.plot(x, [teacher_history[i][0] for i in range(epochs)], label='teacher')
plt.xlim(1, args.Epoch)
#plt.xticks(x)  # 设置x刻度
plt.xlabel('Epoch number')
plt.ylabel('Loss')
plt.title('Test loss')
plt.legend()
plt.show()
'''