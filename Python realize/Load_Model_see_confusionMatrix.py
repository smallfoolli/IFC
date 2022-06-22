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
def randomSeed():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
#加载数据集
data_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5)),

    ])
train_set = ImageFolder(args.Source_data_path1, transform=data_transform)
test_set = ImageFolder(args.Source_data_path2, transform=data_transform)
randomSeed()
# train_set = torch.load(args.Dataset_save_path + '/train_set_p2.pt')
# test_set = torch.load(args.Dataset_save_path + '/test_set_s1_p2.pt')
# train_set = torch.load(args.Dataset_save_path + '/train_set_p2_randomChange.pt')
# test_set = torch.load(args.Dataset_save_path + '/test_set_s1_p2randomChange.pt')
# train_set = torch.load(args.Dataset_save_path + '/train_set_s1_1.5m.pt')
# test_set = torch.load(args.Dataset_save_path + '/test_set_s1_1.5m.pt')

train_loader = Data.DataLoader(dataset=train_set, batch_size=args.BatchSize, shuffle=True)
test_loader = Data.DataLoader(dataset=test_set, batch_size=args.BatchSize, shuffle=True)


# 加载所有模型
# teacher_model1 = torch.load(args.Model_save_path + '/Trained teacher3 model on data2 cross scenario')
# teacher_model2 = torch.load(args.Model_save_path + '/Trained student9 model with KD_LR0006_BS8 in data3 Cross person T150 A06')
# teacher_model3 = torch.load(args.Model_save_path + '/Trained student9 model with KD_LR0006_BS8 in data3 Cross person T150 A07')
# teacher_model4 = torch.load(args.Model_save_path + '/Trained student9 model with KD_LR0006_BS8 in data3 Cross person T150 A08')
# teacher_model5 = torch.load(args.Model_save_path + '/Trained teacher3 model on data2 cross scenario')
# # 上面为测试运行时间加载模型

# teacher_model1 = torch.load(args.Model_save_path + '/Trained student9 model with KD_LR0001 in data3 Cross person T20 A08')
# teacher_model2 = torch.load(args.Model_save_path + '/Trained student9 model with KD_LR0003 in data3 Cross person T20 A08')
# teacher_model3 = torch.load(args.Model_save_path + '/Trained student9 model with KD_LR0008 in data3 Cross person T20 A08')
# teacher_model4 = torch.load(args.Model_save_path + '/Trained student9 model with KD_LR001 in data3 Cross person T20 A08')
# teacher_model5 = torch.load(args.Model_save_path + '/Trained student9 model with KD_LR003 in data3 Cross person T20 A08')
# teacher_model6 = torch.load(args.Model_save_path + '/Trained student9 model with KD_LR006 in data3 Cross person T20 A08')
# teacher_model7 = torch.load(args.Model_save_path + '/Trained student9 model with KD_LR01 in data3 Cross person T20 A08')
# teacher_model8 = torch.load(args.Model_save_path + '/Trained student9 model with KD_LR0001 in data3 Cross person T20 A08')
teacher_model1 = torch.load(args.Model_save_path + '/Trained student9 model with KD_LR0006_BS8 in data3 Cross person T150 A05')
teacher_model2 = torch.load(args.Model_save_path + '/Trained student9 model with KD_LR0006_BS8 in data3 Cross person T150 A06')
teacher_model3 = torch.load(args.Model_save_path + '/Trained student9 model with KD_LR0006_BS8 in data3 Cross person T150 A07')
teacher_model4 = torch.load(args.Model_save_path + '/Trained student9 model with KD_LR0006_BS8 in data3 Cross person T150 A08')
teacher_model5 = torch.load(args.Model_save_path + '/Trained student9 model with KD_LR0006_BS8 in data3 Cross person T150 A09')
# teacher_model6 = torch.load(args.Model_save_path + '/Trained student9 model with KD_LR006 in data3 Cross person T20 A08')
# teacher_model7 = torch.load(args.Model_save_path + '/Trained student9 model with KD_LR01 in data3 Cross person T20 A08')
#加载三个模型，分别测试并输出混淆矩阵。
# teacher_model = torch.load(args.Model_save_path + '/Trained teacher3 model on data2 cross scenario')
# student_model = torch.load(args.Model_save_path + '/Trained student9 model without Infor in data2 Cross scenario')
# student_with_KD_model_ = torch.load(args.Model_save_path + '/Trained student9 model with KD in data2 Cross scenario T75 A09')
# student_model_pretrained = torch.load(args.Model_save_path + '/Trained KDstudent9_versatile model on data2 cross scenario')
# #
# student_model_FullInfo = torch.load(args.Model_save_path + '/Trained student9 model with Full Information in data2 Cross scenario T125 A06')


# teacher_model = torch.load(args.Model_save_path + '/Trained teacher3 model on data3 cross person')
# student_model = torch.load(args.Model_save_path + '/Trained student9 model without Infor in data3 Cross person')
# student_with_KD_model_ = torch.load(args.Model_save_path + '/Trained student9 model with KD in data3 Cross person T20 A08')
# student_model_pretrained = torch.load(args.Model_save_path + '/Trained KDstudent9_versatile model on data3 cross person')
# # student_model_pretrained = torch.load(args.Model_save_path + '/Trained KDstudent9_versatile model on data3 cross person LR LRW change')
# #
# student_model_FullInfo = torch.load(args.Model_save_path + '/Trained student9 model with  Full Information in data3 Cross person T75 A05')


# teacher_model = torch.load(args.Model_save_path + '/Trained teacher model on datasets24Cross')
# student_model = torch.load(args.Model_save_path + '/Trained student8 model on dataset24K5 LR0.00006 BS32')
# # student_with_KD_model = torch.load(args.Model_save_path + '/Trained KD student2_Optim model T1 A09 on p2')
# student_with_KD_model_ = torch.load(args.Model_save_path + '/Trained KD student8_Optim modelK5 T2 A09 on dataset24')
# student_model_pretrained = torch.load(args.Model_save_path + '/Trained student8_versatileWeighted model on dataset24K5 and pretrained model')
# teacher_model = torch.load(args.Model_save_path + '/Trained teacher model on datasetp2p1Cross')
# student_model = torch.load(args.Model_save_path + '/Trained student8 model on datasetp2p1K5 LR0.0016 BS32')
# # student_with_KD_model = torch.load(args.Model_save_path + '/Trained KD student2_Optim model T1 A09 on p2')
# student_with_KD_model_ = torch.load(args.Model_save_path + '/Trained KD student8_Optim modelK5 T100 A06 on datasetp2p1')
# student_model_pretrained = torch.load(args.Model_save_path + '/Trained student8_versatileWeighted model on datasetsp2p1K5 and pretrained model')

# student_model = torch.load(args.Model_save_path + '/Trained student8 model on datasetp2p1 LR0.0001 BS8')
# # student_with_KD_model = torch.load(args.Model_save_path + '/Trained KD student2_Optim model T1 A09 on p2')
# student_with_KD_model_ = torch.load(args.Model_save_path + '/Trained KD student8_Optim model T50 A08 on datasetp2p1')
# student_model_pretrained = torch.load(args.Model_save_path + '/Trained student8_versatileWeighted model on datasetsp2p1 and pretrained model')
# teacher_model = torch.load(args.Model_save_path + '/Trained teacher model on datasetp2')
# # student_model = torch.load(args.Model_save_path + '/Trained student2 model on datasetp2randomChange LR0.0072 BS8')
# student_model = torch.load(args.Model_save_path + '/Trained KD student2_Optim model T1 A00 on p2')
# student_with_KD_model = torch.load(args.Model_save_path + '/Trained KD student2_Optim model T1 A09 on p2')
# student_with_KD_model_ = torch.load(args.Model_save_path + '/Trained KD student2_Optim model T125 A09 on p2')
# student_model_pretrained = torch.load(args.Model_save_path + '/Trained student2_versatileWeighted model on datasetp2 and pretrained model')
#student_model_pretrained = torch.load(args.Model_save_path + '/Trained student_versatile model on datasetp2 and trained model')
#student_model_pretrained_delta0_g2 = torch.load(args.Model_save_path + '/Trained student_versatile_delta0_g2 model on datasetp2 and trained model')



# student_model = torch.load(args.Model_save_path + '/Trained student6 model on datasets115 LR0.0012 BS16')
# student_model_filter_versatile = torch.load(args.Model_save_path + \
#                                             '/Trained student6_versatile model on datasets115 and notrained model')
# student_with_KD_model = torch.load(args.Model_save_path + '/Trained KD student5_Optim model T2 A04 on s115')
# teacher_model = torch.load(args.Model_save_path + '/Trained teacher model')
# student_model = torch.load(args.Model_save_path + '/Trained KD student2_Optim model T1 A00 on s115')
# student_with_KD_model = torch.load(args.Model_save_path + '/Trained KD student2_Optim model T1 A07 on s115')
# student_with_KD_model_ = torch.load(args.Model_save_path + '/Trained KD student2_Optim model T50 A07 on s115')
# student_model_pretrained = torch.load(args.Model_save_path + '/Trained student2_versatileWeighted model on datasets115 and pretrained model')
#student_model_pretrained = torch.load(args.Model_save_path + '/Trained student_versatile model on datasets115 and trained model')
'''
models = [(teacher_model, 'teacher_model'), (student_model, 'student_model'), (student_with_KD_model, 'student_with_KD_model'),
          (student_with_KD_model_, 'student_with_KD_model_'), (student_model_pretrained, 'student_with_KD_model_Versetile'),
          (student_model_pretrained_delta0_g2, 'student_with_KD_model_Versetile_delta0_g2')]
'''
# models = [(teacher_model, 'teacher_model'), (student_model, 'student_model'),
#           (student_with_KD_model_, 'student_with_KD_model_')]


# models = [(teacher_model, 'teacher_model'), (student_model, 'student_model'),
#           (student_with_KD_model_, 'student_with_KD_model_'), (student_model_pretrained, 'student_with_KD_model_Versetile'),
#           (student_model_FullInfo, 'student_model_FullInfo')]

models = [(teacher_model1, 'teacher_model1'), (teacher_model2, 'teacher_model2'),
          (teacher_model3, 'teacher_model3'), (teacher_model4, 'teacher_model4'),
          (teacher_model5, 'teacher_model5')]
# models = [(teacher_model1, 'teacher_model1'), (teacher_model2, 'teacher_model2'),
#           (teacher_model3, 'teacher_model3'), (teacher_model4, 'teacher_model4'),
#           (teacher_model5, 'teacher_model5'), (teacher_model6, 'teacher_model6'), (teacher_model7, 'teacher_model7')]
# models = [(teacher_model, 'teacher_model'), (student_model, 'student_model'), (student_with_KD_model, 'student_with_KD_model'),
#           (student_with_KD_model_, 'student_with_KD_model_'), (student_model_pretrained, 'student_with_KD_model_Versetile')]
# models = [(teacher_model, 'teacher_model'), (student_model, 'student_model'), (student_model_filter_versatile, 'student_with_versatile_model')]
def test_model(model, device, test_loader, figure_title):
    # start_time = time.time()
    randomSeed()
    model.eval()
    test_loss = 0
    correct = 0
    true_label = []
    pred_label = []
    # start_time = time.time()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            true_label.append(target)
            pred_label.append(pred)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    # finish_time = time.time()
    # total_time = finish_time - start_time
    # del start_time
    # del finish_time
    print('测试总用时长为：{}，单个样本所用时长为：{}'.format(total_time, total_time / len(test_loader.dataset)))
    #plt.figure()
    pre_total = torch.cat(pred_label, dim=0)
    true_total = torch.cat(true_label, dim=0)
    # cf.cm_plot(true_total.cpu(), pre_total.cpu())
    # # plt.title('Confusion matrix of ' + figure_title)
    print('\nTest: average loss: {:.4f}, accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    #return test_loss, correct / len(test_loader.dataset)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for model in models:
    test_model(model[0], device, test_loader, figure_title=model[1])
plt.show()