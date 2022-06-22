import pandas as pd
import torch
import matplotlib.pyplot as plt
import argparse
from thop import profile
from torchstat import stat
import torchsummary
from ptflops import get_model_complexity_info
from torch.autograd import Variable
from models.Student_net_versatile_feature_map_Combination import VConv2d
# from models.Student_net_versatile_feature_map_Combination_Norm import VConv2d
import numpy as np
parser = argparse.ArgumentParser(description='Hyperparameter')
parser.add_argument('--Epoch', type=int, default=50, help='The epoch')
parser.add_argument('--input_image_size', type=int, default=96, help='The input_image_size')
parser.add_argument('--Model_save_path', type=str, default='./models/', help='The Trained model save path')
parser.add_argument('--Result_save_path', type=str, default='./result/', help='The result save path')
args = parser.parse_args()
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_params(model, input_size=224):
    # param_sum = 0
    with open('models.txt', 'w') as fm:
        fm.write(str(model))

    # 计算模型的计算量
    calc_flops(model, input_size)

    # 计算模型的参数总量
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    # for p in model_parameters:
    #     print(p.size())
    params = sum([np.prod(p.size()) for p in model_parameters])

    print('The network has {} params.'.format(params))


# 计算模型的计算量
def calc_flops(model, input_size):
    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups) * (
            2 if multiply_adds else 1)
        bias_ops = 1 if self.bias is not None else 0

        # params = output_channels * (kernel_ops + bias_ops)      #bias_ops可以忽略
        params = output_channels * kernel_ops  # bias_ops可以忽略
        flops = batch_size * params * output_height * output_width

        list_conv.append(flops)

    def conv_hookUser(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
        #多样化filter时调用，特征图相加的暂时省略
        for k in range(1, self.kernel_size[0] + 1, 2):
            kernel_ops = k * k * (self.in_channels / self.groups) * (
                2 if multiply_adds else 1)
            bias_ops = 1 if self.bias is not None else 0
            #https://www.jianshu.com/p/b1ceaa7effa8
            # params = output_channels * (kernel_ops + bias_ops)      #bias_ops可以忽略
            params = output_channels * kernel_ops  # bias_ops可以忽略
            flops = batch_size * params * output_height * output_width

            list_conv.append(flops)
        #加权带来的运算量，定义一次乘法一次加法表示一个flop
        flops_w = batch_size * len([i for i in range(1, self.kernel_size[0] + 1, 2)]) * output_height * output_width
        list_conv.append(flops_w)

    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()

        # flops = batch_size * (weight_ops + bias_ops)
        flops = batch_size * weight_ops
        list_linear.append(flops)

    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement())

    def relu_hook(self, input, output):
        # print(input[0].shape)
        # print(input[0].nelement())
        list_relu.append(input[0].nelement())

    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        # params = output_channels * (kernel_ops + bias_ops)
        params = output_channels * kernel_ops
        flops = batch_size * params * output_height * output_width

        list_pooling.append(flops)

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)        #https://www.runoob.com/python/python-func-isinstance.html
            if isinstance(net, VConv2d):
                net.register_forward_hook(conv_hookUser)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                net.register_forward_hook(pooling_hook)
            return
        for c in childrens:
            foo(c)

    multiply_adds = False
    list_conv, list_bn, list_relu, list_linear, list_pooling = [], [], [], [], []
    foo(model)
    input = torch.cuda.FloatTensor(torch.rand(2, 3, input_size, input_size).cuda())
    # if '0.4.' in torch.__version__:
    #     if assets.USE_GPU:
    #         input = torch.cuda.FloatTensor(torch.rand(2, 3, input_size, input_size).cuda())
    #     else:
    #         input = torch.FloatTensor(torch.rand(2, 3, input_size, input_size))
    # else:
    #     input = Variable(torch.rand(2, 3, input_size, input_size), requires_grad=True)
    _ = model(input)

    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling))
    print('  + Number of FLOPs: %.2f' % (total_flops / 2))
    # print('  + Number of FLOPs: %.2fM' % (total_flops / 1e6 / 2))


#加载三个模型，分别计算其参数量以及计算量，然后存储数据。
#teacher_model = torch.load(args.Model_save_path + '/Trained teacher_versatile model on datasets115')
# teacher_model = torch.load(args.Model_save_path + '/Trained teacher model')
# teacher_model_pretrained = torch.load(args.Model_save_path + '/Trained teacher_versatile model on datasets115 and trained model')
# student_model_pretrained = torch.load(args.Model_save_path + '/Trained student_versatile model on datasets115 and trained model')
# student_model_pretrained_ = torch.load(args.Model_save_path + '/Trained student_versatile model on datasetp2 and trained model')
# student_model_pretrained_delta0_g2 = torch.load(args.Model_save_path + '/Trained student_versatile_delta0_g2 model on datasetp2 and trained model')
'''
teacher_model = torch.load(args.Model_save_path + '/Trained teacher model on datasetp2')
student_model = torch.load(args.Model_save_path + '/Trained KD student2_Optim model T1 A00 on p2')
student_with_KD_model = torch.load(args.Model_save_path + '/Trained KD student2_Optim model T1 A09 on p2')
student_with_KD_model_ = torch.load(args.Model_save_path + '/Trained KD student2_Optim model T125 A09 on p2')
'''
'''

params = teacher_model.state_dict() #获得模型的原始状态以及参数。
for k, v in params.items():
    print(k) #只打印key值，不打印具体参数。
'''
# teacher_model = torch.load(args.Model_save_path + '/Trained teacher model on datasetp2')
# student_model = torch.load(args.Model_save_path + '/Trained KD student2_Optim model T1 A00 on p2')
# student_with_KD_model = torch.load(args.Model_save_path + '/Trained KD student2_Optim model T1 A09 on p2')
# student_with_KD_model_ = torch.load(args.Model_save_path + '/Trained KD student2_Optim model T125 A09 on p2')
# student_model_pretrained = torch.load(args.Model_save_path + '/Trained student2_versatileWeighted model on datasetp2 and pretrained model')
#student_model_pretrained = torch.load(args.Model_save_path + '/Trained student_versatile model on datasetp2 and trained model')
# models = [teacher_model, teacher_model_pretrained, student_model_pretrained, student_model_pretrained_,
#           student_model_pretrained_delta0_g2]
# teacher_model = torch.load(args.Model_save_path + '/Trained teacher model')
# student_model = torch.load(args.Model_save_path + '/Trained KD student2_Optim model T1 A00 on s115')
# student_with_KD_model = torch.load(args.Model_save_path + '/Trained KD student2_Optim model T1 A07 on s115')
# student_with_KD_model_ = torch.load(args.Model_save_path + '/Trained KD student2_Optim model T50 A07 on s115')
# student_model_pretrained = torch.load(args.Model_save_path + '/Trained student2_versatileWeighted model on datasets115 and pretrained model')

#
# teacher_model = torch.load(args.Model_save_path + '/Trained teacher model on datasetp2p1Cross')
# student_model = torch.load(args.Model_save_path + '/Trained student8 model on datasetp2p1K5 LR0.0016 BS32')
# # student_with_KD_model = torch.load(args.Model_save_path + '/Trained KD student2_Optim model T1 A09 on p2')
# student_with_KD_model_ = torch.load(args.Model_save_path + '/Trained KD student8_Optim modelK5 T100 A06 on datasetp2p1')
# student_model_pretrained = torch.load(args.Model_save_path + '/Trained student8_versatileWeighted model on datasetsp2p1K5 and pretrained model')


# teacher_model = torch.load(args.Model_save_path + '/Trained teacher3 model on data2 cross scenario')
# student_model = torch.load(args.Model_save_path + '/Trained student9 model without Infor in data2 Cross scenario')
# student_with_KD_model_ = torch.load(args.Model_save_path + '/Trained student9 model with KD in data2 Cross scenario T75 A09')
# student_model_pretrained = torch.load(args.Model_save_path + '/Trained KDstudent9_versatile model on data2 cross scenario')


teacher_model = torch.load(args.Model_save_path + '/Trained teacher3 model on data3 cross person')
student_model = torch.load(args.Model_save_path + '/Trained student9 model without Infor in data3 Cross person')
student_with_KD_model_ = torch.load(args.Model_save_path + '/Trained student9 model with KD in data3 Cross person T20 A08')
student_model_pretrained = torch.load(args.Model_save_path + '/Trained KDstudent9_versatile model on data3 cross person')
student_model_FullInfo = torch.load(args.Model_save_path + '/Trained student9 model with  Full Information in data3 Cross person T75 A05')

# teacher_model = torch.load(args.Model_save_path + '/Trained teacher model on datasets24Cross')
# student_model = torch.load(args.Model_save_path + '/Trained student8 model on dataset24K5 LR0.00006 BS32')
# student_with_KD_model_ = torch.load(args.Model_save_path + '/Trained KD student8_Optim modelK5 T2 A09 on dataset24')
# student_model_pretrained = torch.load(args.Model_save_path + '/Trained student8_versatileWeighted model on dataset24K5 and pretrained model')
#

models = [teacher_model, student_model,
          student_with_KD_model_, student_model_pretrained, student_model_FullInfo]
para_flops = []
Different_Net_Comparation = {}
input_image = torch.randn(1, 3, args.input_image_size, args.input_image_size).cuda()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# for model in models:
#     flops, params = profile(model.to(device), inputs=(input_image,))
#     para_flops.append((params, flops))
#
# Different_Net_Comparation['Para'] = [item[0] for item in para_flops]
# Different_Net_Comparation['Flop'] = [item[1] for item in para_flops]
#
# print(Different_Net_Comparation)
# https://github.com/sksq96/pytorch-summary
# from torchsummary import summary
# summary(your_model, input_size=(channels, H, W))
# for model in models:
#     torchsummary.summary(model, input_size=(3, args.input_image_size, args.input_image_size))
#     print('parameters_count:', count_parameters(model))

for model in models:
    count_params(model, args.input_image_size)
    # print('parameters_count:', count_parameters(model))