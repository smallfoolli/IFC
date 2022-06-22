# coding=utf-8
import pandas as pd
import torch
import matplotlib.pyplot as plt
import confusion as cf
import argparse
import torch.utils.data as Data
import torch.nn.functional as F
from models.Teacher_Class import TeacherNet as TeacherModel


parser = argparse.ArgumentParser(description='Hyperparameter')
parser.add_argument('--input_image_size', type=int, default=96, help='The input_image_size')
#parser.add_argument('--LR', type=float, default=[0.00006, 0.00008, 0.0001, 0.0003, 0.0005, 0.0007, 0.0009], help='The learning rate')
#parser.add_argument('--LR', type=float, default=[0.0003, 0.0005, 0.0007, 0.0009, 0.0010, 0.0012, 0.0014, 0.0016, 0.0018,
#                                                 0.0020, 0.0022, 0.0024, 0.0026], help='The learning rate')
parser.add_argument('--LR', type=float, default=[0.00006, 0.00008, 0.0001, 0.0003, 0.0005, 0.0007, 0.0009, 0.0010,
                                                 0.0012, 0.0014, 0.0016, 0.0018,
                                                 0.0020, 0.0022, 0.0024, 0.0026], help='The learning rate')
parser.add_argument('--BatchSize', type=int, default=[8, 16, 32], help='The BatchSize')
parser.add_argument('--Epoch', type=int, default=30, help='The epoch')
parser.add_argument('--Dataset_save_path', type=str, default='./Dataset/', help='The Dataset_save_path')
parser.add_argument('--Model_save_path', type=str, default='./models/', help='The Trained model save path')
parser.add_argument('--Result_save_path', type=str, default='./results/', help='The result save path')
args = parser.parse_args()

Accuracy_change_line = {}
Loss_change_line = {}
#开始读取数据，绘制相应的分析图
# teacher_data = pd.read_csv(args.Result_save_path + 'Teacher results_LR_Batchsize on datasetp2.csv')
# teacher_data = pd.read_csv(args.Result_save_path + 'Student6 results_LR_Batchsize on dataset_s115.csv')
# teacher_data = pd.read_csv(args.Result_save_path + 'Student2 results_LRW_search1 on dataset_p2.csv')
# teacher_data = pd.read_csv(args.Result_save_path + 'Student2 results_LRW_search1_randomState on dataset_p2.csv')
# teacher_data = pd.read_csv(args.Result_save_path + 'Student8 results_LRW_search1_randomStateK5 on datasetp2p1.csv')
teacher_data = pd.read_csv(args.Result_save_path + 'Student8 results_LRW_search1_randomStateK5 on dataset24.csv')
# df_result.to_csv(args.Result_save_path + 'Student8 results_LRW_search1_randomStateK5 on dataset24.csv', index=False, sep=',')
# LRW = [item / 100 for item in range(1, 51)]
# LRW = [item / 1000 for item in range(1, 350, 5)]
LRW = [item / 4000 for item in range(1, 401, 4)]
# LRW = [item / 1000 for item in range(1, 501, 4)]
for LR in LRW:
# for LR in args.LR:
#     Accuracy_change_line['LRW= ' + str(LR)] = \
#         teacher_data['LRW= ' + str(LR) + ' Acc'][args.Epoch - 20:]
    Accuracy_change_line['LRW= ' + str(LR)] = \
        teacher_data['LRW= ' + str(LR) + ' Acc']
    Loss_change_line['LRW= ' + str(LR)] = \
        teacher_data['LRW= ' + str(LR) + ' Loss']
    # for Batchsize in args.BatchSize:
    #     Accuracy_change_line['LR= ' + str(LR) + ' BS= ' + str(Batchsize)] = \
    #         teacher_data['LR= ' + str(LR) + ' BS= ' + str(Batchsize) + ' Acc'][args.Epoch-20:]
    #     Loss_change_line['LR= ' + str(LR) + ' BS= ' + str(Batchsize)] = \
    #         teacher_data['LR= ' + str(LR) + ' BS= ' + str(Batchsize) + ' Loss']

Accuracy_statistic = pd.DataFrame(Accuracy_change_line)
print(Accuracy_statistic.describe())
Accuracy_statistic_result = Accuracy_statistic.describe()

Loss_statistic = pd.DataFrame(Loss_change_line)


Accuracy_statistic.boxplot(showfliers=True, grid=True, showmeans=True)     # 这里，pandas自己有处理的过程。
#showfliers为异常点是否显示
plt.xticks(rotation=45)
plt.ylabel('Accuracy')
plt.xlabel('Learning rate featureMap')    # 我们设置横纵坐标的标题。
plt.title('Accuracy distribution under different hyperparameter')

plt.figure(2)
Accuracy_LR_change_line = []
temBS = 32
for LR in LRW:

    Accuracy_LR_change_line.append((LR,
                                    teacher_data['LRW= ' + str(LR) + ' Acc'][args.Epoch-20:].mean()))

print(Accuracy_LR_change_line)

plt.plot(LRW, [value[1] for value in Accuracy_LR_change_line])
plt.ylabel('Accuracy')
plt.xlabel('Learning rate')
plt.title('Accuracy  under different learning rate')

# plt.figure(3)
# epochs = [i+1 for i in range(args.Epoch)]
# for LR in args.LR:
#     for Batchsize in args.BatchSize:
#         plt.plot(epochs, Loss_change_line['LR= ' + str(LR) + ' BS= ' + str(Batchsize)],
#                  label='LR= ' + str(LR) + ' BS= ' + str(Batchsize))
#
# plt.ylabel('Loss')
# plt.xlabel('Epoch')    # 我们设置横纵坐标的标题。
# #plt.title('Accuracy  under different learning rate')
# plt.xlim(1, args.Epoch)
# plt.legend()
plt.show()