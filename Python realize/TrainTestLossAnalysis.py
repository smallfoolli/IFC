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
# parser.add_argument('--LR', type=float, default=[0.00006, 0.00008, 0.0001, 0.0003, 0.0005, 0.0007, 0.0009], help='The learning rate')
#parser.add_argument('--LR', type=float, default=[0.0003, 0.0005, 0.0007, 0.0009, 0.0010, 0.0012, 0.0014, 0.0016, 0.0018,
#                                                 0.0020, 0.0022, 0.0024, 0.0026], help='The learning rate')
# parser.add_argument('--LR', type=float, default=[0.00006, 0.00008, 0.0001, 0.0003, 0.0005, 0.0007, 0.0009, 0.0010,
#                                                  0.0012, 0.0014, 0.0016, 0.0018,
#                                                  0.0020, 0.0022, 0.0024, 0.0026], help='The learning rate')
# parser.add_argument('--LR', type=float, default=[0.00006, 0.0003, 0.0009, 0.0014, 0.0022, 0.0026,
#                                                  0.0034, 0.0042, 0.0052, 0.0062, 0.0072, 0.0082], help='The learning rate')

# parser.add_argument('--LR', type=float, default=[0.00006, 0.00008, 0.0001, 0.0003, 0.0005, 0.0007, 0.0009, 0.0010,
#                                                  0.002, 0.004, 0.006, 0.008], help='The learning rate')
# parser.add_argument('--LR', type=float, default=[0.0010, 0.006, 0.008, 0.03], help='The learning rate')
# parser.add_argument('--LR', type=float, default=[0.0006, 0.0008, 0.0010, 0.003], help='The learning rate')
# parser.add_argument('--LR', type=float, default=[0.0001], help='The learning rate')
parser.add_argument('--LR', type=float, default=[0.0001, 0.0006, 0.0008, 0.0010, 0.003,  0.005], help='The learning rate')
#
# parser.add_argument('--LR', type=float, default=[0.0001], help='The learning rate')
# parser.add_argument('--BatchSize', type=int, default=[8, 16, 32, 64, 128, 256], help='The BatchSize')
parser.add_argument('--BatchSize', type=int, default=[8], help='The BatchSize')
parser.add_argument('--Epoch', type=int, default=55, help='The epoch')
parser.add_argument('--Dataset_save_path', type=str, default='./Dataset/', help='The Dataset_save_path')
parser.add_argument('--Model_save_path', type=str, default='./models/', help='The Trained model save path')
parser.add_argument('--Result_save_path', type=str, default='./result/', help='The result save path')
args = parser.parse_args()

Accuracy_change_line = {}
# Loss_change_line = {}
Train_Loss_change_line = {}
Test_Loss_change_line = {}
#?????????????????????????????????????????????
# teacher_data = pd.read_csv(args.Result_save_path + 'Teacher results_LR_Batchsize on datasetp2.csv')
# teacher_data = pd.read_csv(args.Result_save_path + 'Student6 results_LR_Batchsize on dataset_s115.csv')
# teacher_data = pd.read_csv(args.Result_save_path + 'Student8 results_LR_Batchsize on dataset_24K5.csv')
# teacher_data = pd.read_csv(args.Result_save_path + 'Teacher results_LR_Batchsize on 24Cross.csv')
# teacher_data = pd.read_csv(args.Result_save_path + 'Student8 results_LR_Batchsize on dataset_p2p1K5.csv')

# teacher_data = pd.read_csv(args.Result_save_path + 'Teacher results_LR_Batchsize on data1 normal.csv')
# teacher_data = pd.read_csv(args.Result_save_path + 'Teacher results_LR_Batchsize on data2 cross scenerio.csv')
# teacher_data = pd.read_csv(args.Result_save_path + 'Teacher2 results_LR_Batchsize8 on data2 cross scenerio.csv')
# teacher_data = pd.read_csv(args.Result_save_path + 'Teacher2 results_LR_FineSearch_Batchsize8 on data2 cross scenerio.csv')
# teacher_data = pd.read_csv(args.Result_save_path + 'Teacher2 results_LR_Batch vari FineSearch on data2 cross scenerio.csv')
# teacher_data = pd.read_csv(args.Result_save_path + 'Teacher2 results_LR_optim Search on data2 cross scenerio.csv')
# teacher_data = pd.read_csv(args.Result_save_path + 'Student8 results_LRFix_optim search on data2 cross scenerio.csv')
teacher_data = pd.read_csv(args.Result_save_path + 'Teacher3 results_LR_optim search on data3 cross person.csv')
# teacher_data = pd.read_csv(args.Result_save_path + 'Student8 results_LRVar_optim search on data3 cross person.csv')
# teacher_data = pd.read_csv(args.Result_save_path + 'Teacher3 results_LR_optim BSsearch on data3 cross person.csv')
for LR in args.LR:
    for Batchsize in args.BatchSize:
        # Accuracy_change_line['LR= ' + str(LR) + ' BS= ' + str(Batchsize)] = \
        #     teacher_data['LR= ' + str(LR) + ' BS= ' + str(Batchsize) + ' Acc'][args.Epoch-20:]
        Accuracy_change_line['LR= ' + str(LR) + ' BS= ' + str(Batchsize)] = \
            teacher_data['LR= ' + str(LR) + ' BS= ' + str(Batchsize) + ' Acc']
        # Loss_change_line['LR= ' + str(LR) + ' BS= ' + str(Batchsize)] = \
        #     teacher_data['LR= ' + str(LR) + ' BS= ' + str(Batchsize) + ' Loss']

        # Train_Loss_change_line['LR= ' + str(LR) + ' BS= ' + str(Batchsize)] = \
        #     [float(dataT[7:13]) for dataT in teacher_data['LR= ' + str(LR) + ' BS= ' + str(Batchsize) + ' TrainLoss']]
        Train_Loss_change_line['LR= ' + str(LR) + ' BS= ' + str(Batchsize)] = \
            teacher_data['LR= ' + str(LR) + ' BS= ' + str(Batchsize) + ' TrainLoss']
        # print([dataT for dataT in teacher_data['LR= ' + str(LR) + ' BS= ' + str(Batchsize) + ' TrainLoss']])
        # print([float(dataT[7:13]) for dataT in teacher_data['LR= ' + str(LR) + ' BS= ' + str(Batchsize) + ' TrainLoss']])
        Test_Loss_change_line['LR= ' + str(LR) + ' BS= ' + str(Batchsize)] = \
            teacher_data['LR= ' + str(LR) + ' BS= ' + str(Batchsize) + ' TestLoss']
        # Accuracy_change_line['LR= ' + str(LR) + ' BS= ' + str(Batchsize)] = \
        #     teacher_data['LR= ' + str(LR) + ' BS= ' + str(Batchsize) + ' Acc']
        # Loss_change_line['LR= ' + str(LR) + ' BS= ' + str(Batchsize)] = \
        #     teacher_data['LR= ' + str(LR) + ' BS= ' + str(Batchsize) + ' Loss']

# Accuracy_statistic = pd.DataFrame(Accuracy_change_line)
# print(Accuracy_statistic.describe())
# Accuracy_statistic_result = Accuracy_statistic.describe()
#
# Loss_statistic = pd.DataFrame(Loss_change_line)
#
#
# Accuracy_statistic.boxplot(showfliers=True, grid=True, showmeans=True)     # ?????????pandas???????????????????????????
# #showfliers????????????????????????
# plt.xticks(rotation=45)
# plt.ylabel('Accuracy')
# plt.xlabel('Learning rate, BatchSize')    # ????????????????????????????????????
# plt.title('Accuracy distribution under different hyperparameter')
#
# plt.figure(2)
# Accuracy_LR_change_line = []
# temBS = 32
# for LR in args.LR:
#
#     Accuracy_LR_change_line.append((LR,
#                                     teacher_data['LR= ' + str(LR) + ' BS= ' + str(temBS) + ' Acc'][args.Epoch-20:].mean()))
#
# print(Accuracy_LR_change_line)
#
# plt.plot(args.LR, [value[1] for value in Accuracy_LR_change_line])
# plt.ylabel('Accuracy')
# plt.xlabel('Learning rate')
# plt.title('Accuracy  under different learning rate')

# plt.figure(3)
epochs = [i+1 for i in range(args.Epoch)]
count = 1
for LR in args.LR:
    for Batchsize in args.BatchSize:
        plt.figure(count)
        plt.subplot(2, 2, 1)
        plt.plot(epochs, Train_Loss_change_line['LR= ' + str(LR) + ' BS= ' + str(Batchsize)],
                 label='LR= ' + str(LR) + ' BS= ' + str(Batchsize))
        plt.ylabel('Train Loss')
        plt.xlabel('Epoch')  # ????????????????????????????????????
        # plt.title('Accuracy  under different learning rate')
        plt.xlim(1, args.Epoch)
        plt.title('LR= ' + str(LR) + ' BS= ' + str(Batchsize))

        plt.subplot(2, 2, 2)
        plt.plot(epochs, Test_Loss_change_line['LR= ' + str(LR) + ' BS= ' + str(Batchsize)],
                 label='LR= ' + str(LR) + ' BS= ' + str(Batchsize))
        plt.ylabel('Test Loss')
        plt.xlabel('Epoch')  # ????????????????????????????????????
        # plt.title('Accuracy  under different learning rate')
        plt.xlim(1, args.Epoch)
        plt.title('LR= ' + str(LR) + ' BS= ' + str(Batchsize))

        plt.subplot(2, 1, 2)
        plt.plot(epochs, Accuracy_change_line['LR= ' + str(LR) + ' BS= ' + str(Batchsize)],
                 label='LR= ' + str(LR) + ' BS= ' + str(Batchsize))
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')  # ????????????????????????????????????
        # plt.title('Accuracy  under different learning rate')
        plt.xlim(1, args.Epoch)
        plt.title('LR= ' + str(LR) + ' BS= ' + str(Batchsize))

        # plt.legend()
        plt.show()

# plt.ylabel('Loss')
# plt.xlabel('Epoch')    # ????????????????????????????????????
# #plt.title('Accuracy  under different learning rate')
# plt.xlim(1, args.Epoch)
# plt.legend()
# plt.show()