import math
import torch
import random
import time
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import torch.utils.data as Data
from models.Student_Class import StudentNet9 as StudentModel

import pandas as pd



import argparse
parser = argparse.ArgumentParser(description='Hyperparameter')
parser.add_argument('--input_image_size', type=int, default=96, help='The input_image_size')
# parser.add_argument('--LR', type=float, default=[0.0001, 0.0006, 0.0008, 0.0010, 0.003,  0.005], help='The learning rate')
parser.add_argument('--LR', type=int, default=0.0006, help='The learning rate')
parser.add_argument('--BatchSize', type=int, default=8, help='The BatchSize')
parser.add_argument('--Epoch', type=int, default=55, help='The epoch')
parser.add_argument('--Temperature', type=float, default=[75.0], help='The Temperature')
parser.add_argument('--Alpha', type=float, default=[0.8], help='The Alpha')
parser.add_argument('--Source_data_path', type=str, default='./data/s1_1.5m/', help='The Source data path')
parser.add_argument('--Dataset_save_path', type=str, default='./Dataset/', help='The Dataset_save_path')

parser.add_argument('--Model_save_path', type=str, default='./models/', help='The Trained model save path')
parser.add_argument('--Result_save_path', type=str, default='./result/', help='The result save path')
# parser.add_argument('--Source_data_path2', type=str, default='./data/24Test/', help='The Source data path')
# parser.add_argument('--Source_data_path1', type=str, default='./data/24Train/', help='The Source data path')

# parser.add_argument('--Source_data_path1', type=str, default='./radarDataNew/data2_12/1', help='The Source data path')
# parser.add_argument('--Source_data_path2', type=str, default='./radarDataNew/data2_12/2', help='The Source data path')


parser.add_argument('--Source_data_path1', type=str, default='./radarDataNew/data3_12/train', help='The Source data path')
parser.add_argument('--Source_data_path2', type=str, default='./radarDataNew/data3_12/test', help='The Source data path')


args = parser.parse_args()

#加载数据集
#train_set = torch.load(args.Dataset_save_path + '/train_set_p2.pt')
#test_set = torch.load(args.Dataset_save_path + '/test_set_s1_p2.pt')

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

#torch.manual_seed(0)
#torch.cuda.manual_seed(0)



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

    print('\nTest: average loss: {:.4f}, accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss, correct / len(test_loader.dataset)


def KD_student_main(args, train_loader, test_loader):
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StudentModel(args.input_image_size).to(device)
    # teacher_model = torch.load(args.Model_save_path + '/Trained teacher3 model on data2 cross scenario')
    teacher_model = torch.load(args.Model_save_path + '/Trained teacher3 model on data3 cross person')
    # teacher_model = torch.load(args.Model_save_path + '/Trained teacher model on datasets24Cross')
    # teacher_model = torch.load(args.Model_save_path + '/Trained teacher model on datasetp2p1Cross')
    #teacher_model = torch.load(args.Model_save_path + '/Trained teacher model on datasetp2')
    # teacher_model = torch.load(args.Model_save_path + '/Trained teacher model')
    teacher_model = teacher_model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.LR, weight_decay=1e-5)
    KD_student_history = []
    for temperature in args.Temperature:
        for alpha in args.Alpha:
            #model = StudentModel(args.input_image_size).to(device)
            #optimizer = torch.optim.Adam(model.parameters(), lr=args.LR, weight_decay=1e-5)
            LR = args.LR
            print('T = {} Alpah = {}'.format(temperature, alpha))
            for epoch in range(1, args.Epoch + 1):
                if epoch % 10 == 0:
                    LR = LR / 5
                    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
                KD_train_student(model, teacher_model, device, train_loader, optimizer, epoch, temperature, alpha)
                # loss, acc = KD_test_student(model, device, test_loader)
                #
                # KD_student_history.append((epoch, temperature, alpha, loss, acc))

    #for epoch in range(1, args.Epoch + 1):
        #KD_train_student(model, teacher_model, device, train_loader, optimizer, epoch, temperature, alpha)
        #loss, acc = KD_test_student(model, device, test_loader)

        #KD_student_history.append((loss, acc))

    #torch.save(model.state_dict(), "teacher.pt")
    # torch.save(model, args.Model_save_path + '/Trained KD student8_Optim modelK5 T100 A06 on datasetp2p1')
    # torch.save(model, args.Model_save_path + '/Trained KD student8_Optim modelK5 T2 A09 on dataset24')
    # torch.save(model, args.Model_save_path + '/Trained student8 model with KD in data2 Cross scenario T10 A05')
    # torch.save(model, args.Model_save_path + '/Trained student8 model with KD in data3 Cross person T100 A09')
    # torch.save(model, args.Model_save_path + '/Trained student9 model withlr0006 KD in data2 Cross scenario T20 A08')
    # torch.save(model, args.Model_save_path + '/Trained student9 model without KD_LR0006_BS128 in data3 Cross person')
    return model, KD_student_history


#torch.save(model, './modelstore/model_before_prue')
#model = torch.load('./modelstore/letter_class_feature932515')



start_time = time.time()
student_model, KD_student_history = KD_student_main(args, train_loader, test_loader)
finish_time = time.time()
total_time = finish_time - start_time
print('蒸馏总用时长为：{}'.format(total_time))
#可以存储一下结果，存成CSV或者Excel

# result_total = {'ID': [i+1 for i in range(len(KD_student_history))],
#                 'Epoch': [KD_student_history[i][0] for i in range(len(KD_student_history))],
#                 'KD temperature': [KD_student_history[i][1] for i in range(len(KD_student_history))],
#                 'KD alpha': [KD_student_history[i][2] for i in range(len(KD_student_history))],
#                 'Test Loss': [KD_student_history[i][3] for i in range(len(KD_student_history))],
#                 'Test Accuracy': [KD_student_history[i][4] for i in range(len(KD_student_history))]}
# df_result = pd.DataFrame(result_total)
# print(df_result)
# df_result.to_csv(args.Result_save_path + 'Student6_Optim with KD results T2A02 on s115.csv', index=False, sep=',')
#df_result.to_csv(args.Result_save_path + 'Student2_Optim with KD results TVAV on datasetp2.csv', index=False, sep=',')
# df_result.to_csv(args.Result_save_path + 'Student8_Optim with KD results T2 A09 on dataset24.csv', index=False, sep=',')
# df_result.to_csv(args.Result_save_path + 'Student9 results_LR0002 withKD T20 A08.csv', index=False, sep=',')
# df_result.to_csv(args.Result_save_path + 'Student9 results_LR0006BS128 withoutKD in data3 Cross person.csv', index=False, sep=',')
#
