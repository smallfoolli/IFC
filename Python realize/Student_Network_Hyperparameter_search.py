#coding=utf-8
import pandas as pd
import torch
import matplotlib.pyplot as plt
import confusion as cf
import argparse
import torch.utils.data as Data
import torch.nn.functional as F
from models.Student_Class import StudentNet8 as StudentModel
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder


parser = argparse.ArgumentParser(description='Hyperparameter')
parser.add_argument('--input_image_size', type=int, default=96, help='The input_image_size')
#parser.add_argument('--LR', type=float, default=[0.00006, 0.00008], help='The learning rate')
parser.add_argument('--LR', type=float, default=[0.00006, 0.00008, 0.0001, 0.0003, 0.0005, 0.0007, 0.0009, 0.0010,
                                                 0.0012, 0.0014, 0.0016, 0.0018,
                                                 0.0020, 0.0022, 0.0024, 0.0026], help='The learning rate')
# parser.add_argument('--LR', type=float, default=[0.00006, 0.0003, 0.0009, 0.0014, 0.0022, 0.0026,
#                                                  0.0034, 0.0042, 0.0052, 0.0062, 0.0072, 0.0082], help='The learning rate')
parser.add_argument('--BatchSize', type=int, default=[8, 16, 32], help='The BatchSize')
parser.add_argument('--Epoch', type=int, default=50, help='The epoch')
parser.add_argument('--Dataset_save_path', type=str, default='./Dataset/', help='The Dataset_save_path')
parser.add_argument('--Model_save_path', type=str, default='./models/', help='The Trained model save path')
parser.add_argument('--Result_save_path', type=str, default='./results/', help='The result save path')
parser.add_argument('--Source_data_path2', type=str, default='./data/24Test/', help='The Source data path')
parser.add_argument('--Source_data_path1', type=str, default='./data/24Train/', help='The Source data path')
args = parser.parse_args()

#加载数据集
# train_set = torch.load(args.Dataset_save_path + '/train_set_p2.pt')
# test_set = torch.load(args.Dataset_save_path + '/test_set_s1_p2.pt')
# train_set = torch.load(args.Dataset_save_path + '/train_set_p2_randomChange50.pt')
# test_set = torch.load(args.Dataset_save_path + '/test_set_s1_p2randomChange50.pt')
# train_set = torch.load(args.Dataset_save_path + '/train_set_s1_1.5m.pt')
# test_set = torch.load(args.Dataset_save_path + '/test_set_s1_1.5m.pt')
#加载数据集
data_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5)),

    ])
train_set = ImageFolder(args.Source_data_path1, transform=data_transform)
test_set = ImageFolder(args.Source_data_path2, transform=data_transform)


def train_student(model, device, train_loader, optimizer, epoch):
    model.train()
    trained_samples = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        trained_samples += len(data)
        print("\rTrain epoch %d: %d/%d" %
              (epoch, trained_samples, len(train_loader.dataset)), end='')


def test_student(model, device, test_loader):
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


def student_main(args, train_loader, test_loader, LR):
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StudentModel(args.input_image_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    student_history = []

    for epoch in range(1, args.Epoch + 1):
        train_student(model, device, train_loader, optimizer, epoch)
        loss, acc = test_student(model, device, test_loader)

        student_history.append((loss, acc))

    #torch.save(model.state_dict(), "teacher.pt")
    #torch.save(model, args.Model_save_path + '/Trained teacher model')
    return student_history

#保存整个模型
#torch.save(model, './modelstore/model_before_prue')
#model = torch.load('./modelstore/letter_class_feature932515')
data_result = {}
# 训练教师网络
for LR in args.LR:
    for Batchsize in args.BatchSize:
        train_loader = Data.DataLoader(dataset=train_set, batch_size=Batchsize, shuffle=True)
        test_loader = Data.DataLoader(dataset=test_set, batch_size=Batchsize, shuffle=True)
        #teacher_history = []
        student_history = student_main(args, train_loader, test_loader, LR)
        data_result['LR= ' + str(LR) + ' BS= ' + str(Batchsize) + ' Loss'] = [student_history[i][0] for i in range(args.Epoch)]
        data_result['LR= ' + str(LR) + ' BS= ' + str(Batchsize) + ' Acc'] = [student_history[i][1] for i in range(args.Epoch)]

#teacher_model, teacher_history = teacher_main(args, train_loader, test_loader)
#可以存储一下结果，存成CSV或者Excel
'''
result_total = {'Epoch': [i+1 for i in range(args.Epoch)],
                'Test Loss': [teacher_history[i][0] for i in range(args.Epoch)],
                'Test Accuracy': [teacher_history[i][1] for i in range(args.Epoch)]}

'''
df_result = pd.DataFrame(data_result)
print(df_result)
#df_result.to_csv(args.Result_save_path + 'Student6 results_LR_Batchsize on dataset_s115.csv', index=False, sep=',')
# df_result.to_csv(args.Result_save_path + 'Student8 results_LR_Batchsize on dataset_p2.csv', index=False, sep=',')
df_result.to_csv(args.Result_save_path + 'Student8 results_LR_Batchsize on dataset_24K5.csv', index=False, sep=',')
#画出训练测试曲线
