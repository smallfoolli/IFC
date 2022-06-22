import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import math
import numpy as np
import random

__all__ = [
    'Student',
]
# torch.manual_seed(0)
# torch.cuda.manual_seed(0)
# kernel_size_layer = [5]
kernel_size_layer = [7, 7]
def randomSeed():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            if v % 2 == 0:
                v1 = v * int(np.ceil(3/2))      #use spatial filter versatile to increase features
                conv2d = VConv2d(in_channels, v1, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v1), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v1
            else:
                conv2d = VConv2d(in_channels, v-1, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v-1), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v - 1
    return nn.Sequential(*layers)

def make_layersO(cfg, batch_norm=False, ks=[]):
    layers = []
    in_channels = 3
    indx = 0
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = VConv2d(in_channels, v, kernel_size=ks[indx], padding=1)
            indx += 1
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
cfg = {
    'A': [64, 'M', 128, 'M', 256, 'M', 512, 'M'],
    'B': [9, 'M', 9, 'M'],
    'C': [4, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'F': [6, 'M', 4, 'M'],
    'G': [6, 'M'],
    'H': [6, 'M', 12, 'M'],
}
def Calculate_ImageSize_AfterConvBlocks(input_size, kernel_size=3, stride=1, padding=1):
    conv_size = (input_size - kernel_size + 2 * padding) // stride + 1
    after_pool = math.ceil((conv_size - 2) / 2) + 1
    return after_pool



class VConv2d(nn.modules.conv._ConvNd):
  """
  Versatile Filters
  Paper: https://papers.nips.cc/paper/7433-learning-versatile-filters-for-efficient-convolutional-neural-networks
  """


  def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, delta=0, g=1):
    # torch.manual_seed(0)
    # torch.cuda.manual_seed(0)
    # randomSeed()
    kernel_size = _pair(kernel_size)
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    super(VConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
        False, _pair(0), groups, bias, 'zeros')
    '''
    super(VConv2d, self).__init__(
        in_channels, out_channels, kernel_size, stride, padding, dilation,
        False, _pair(0), groups, bias)
    '''
    self.s_num = int(np.ceil(self.kernel_size[0]/2))  # s in paper
    self.delta = delta  # c-\hat{c} in paper
    self.g = g  # g in paper
    #(1+self.delta/self.g)表示公式8里的n
    #groups代表啥？分组卷积？
    # self.weight = nn.Parameter(torch.Tensor(
    #             int(out_channels/self.s_num/(1+self.delta/self.g)), in_channels // groups, *kernel_size))
    #下面实验多尺度特征图叠加的思路
    self.weight = nn.Parameter(torch.Tensor(
                            int(out_channels), in_channels // groups, *kernel_size))
    # if self.s_num * 2 > out_channels:
    #     self.weight = nn.Parameter(torch.Tensor(
    #                 int(out_channels / (self.s_num - 1)), in_channels // groups, *kernel_size))
    # else:
    #     self.weight = nn.Parameter(torch.Tensor(
    #                     int(out_channels / self.s_num), in_channels // groups, *kernel_size))
    # #out_channels/self.s_num/(1+self.delta/self.g)这里控制了每个二级filter输出的通道数
    # self.weightFeatureMap = nn.Parameter(torch.Tensor([1/self.s_num]*self.s_num))
    #生成一个列表，长度为n，其中每个元素都是1/n
    # self.weightFeatureMap = nn.Parameter(torch.Tensor([0.8, 0.15, 0.05]))
    self.reset_parameters()

  def forward(self, x):
    # torch.manual_seed(0)
    # torch.cuda.manual_seed(0)
    # randomSeed()
    x_list = []
    s_num = self.s_num
    ch_ratio = (1+self.delta/self.g)        #论文里公式8的n
    ch_len = self.in_channels - self.delta      #
    for s in range(s_num):
        for start in range(0, self.delta+1, self.g):
            #weight1的维度意义分别代表输出通道数，输入通道数，kernel_size
            weight1 = self.weight[:, :ch_len, s:self.kernel_size[0]-s, s:self.kernel_size[0]-s]
            if self.padding[0]-s < 0:
                h = x.size(2)
                x1 = x[:, start:start+ch_len, s:h-s, s:h-s]
                padding1 = _pair(1)
            else:
                x1 = x[:, start:start+ch_len, :, :]
                padding1 = _pair(self.padding[0]-s)
            # 卷积并对每种size大小filter的结果进行采样，最后拼接特征图
            tem_feature = F.conv2d(x1, weight1, self.bias, self.stride, padding1, self.dilation, self.groups)
            samp_num = np.ceil(self.out_channels / s_num)
            random_num = random.sample(range(0, self.out_channels), int(samp_num))
            for item in random_num:
                x_list.append(tem_feature[:, item:item+1, :, :])
            # 逐渐切一块偏置进行卷积运算
            # if s_num * 2 > self.out_channels:
            #     x_list.append(F.conv2d(x1, weight1, self.bias[int(
            #         np.ceil(self.out_channels * (s * ch_ratio + start) / (s_num - 1) / ch_ratio)):int(
            #         self.out_channels * (s * ch_ratio + start + 1) / (s_num - 1) / ch_ratio)], self.stride,
            #                            padding1, self.dilation, self.groups))
            # else:
            #     x_list.append(F.conv2d(x1, weight1, self.bias[int(np.ceil(self.out_channels*(s*ch_ratio+start)/s_num/ch_ratio)):int(self.out_channels*(s*ch_ratio+start+1)/s_num/ch_ratio)], self.stride,
            #               padding1, self.dilation, self.groups))
            # print('查看bias的情况属于论文中描述的哪一种')
            # print(self.bias[int(
            #     self.out_channels * (s * ch_ratio + start) / s_num / ch_ratio):int(
            #     self.out_channels * (s * ch_ratio + start + 1) / s_num / ch_ratio)])
            # bias = self.bias.data.reshape(s_num, -1)
            # bias_ = bias.mean(axis=0, keepdim=False)
            #bias不进行共享
            # x_list.append(F.conv2d(x1, weight1, self.bias, self.stride,
            #                        padding1, self.dilation, self.groups))
            # x_list.append(F.conv2d(x1, weight1, self.bias[int(
            #     self.out_channels * (s * ch_ratio + start) / s_num / ch_ratio):int(
            #     self.out_channels * (s * ch_ratio + start + 1) / s_num / ch_ratio)], self.stride,
            #                        padding1, self.dilation, self.groups))

            #bias共享
            # x_list.append(F.conv2d(x1, weight1, bias_, self.stride,
            #                        padding1, self.dilation, self.groups))
            #F.conv2d的使用说明参考下面链接
            #该代码的局限性在于原输出通道数得是偶数的情况，如果是奇数则需要改动的很多
            #https://blog.csdn.net/SHU15121856/article/details/88956545
    #开始高维矩阵叠加，也就是特征图的叠加
    # weight_featureMap = [0.6, 0.4, 0.0]
    # for i in range(len(weight_featureMap)):
    #     x_list[i] *= weight_featureMap[i]
    # print('featureMap的权值为：')
    # print(self.weightFeatureMap)
    # for i in range(len(self.weightFeatureMap)):
    #     x_list[i] *= self.weightFeatureMap[i]
    # x = sum(x_list)
    # 每种filter运算得到的特征图进行拼接，形成下一层的输入
    if len(x_list) == self.out_channels:
        x = torch.cat(x_list, 1)
    else:
        sam_numt = random.sample(range(0, len(x_list)), self.out_channels)
        tem_lis = []
        for item in sam_numt:
            tem_lis.append(x_list[item])
        x = torch.cat(tem_lis, 1)
    #torch.cat除拼接维数dim数值可不同外其余维数数值需相同，方能对齐。
    #https://blog.csdn.net/zhanly19/article/details/96428781
    return x


class Student_net9(nn.Module):
    def __init__(self, features, num_classes=12, init_weights=True, inputImageSize=96, outputImageSize=0,
                 NumberConvBlocks=1):
        super(Student_net9, self).__init__()
        self.features = features
        self.inputImageSize = inputImageSize
        self.outputImageSize = outputImageSize
        self.outputImageSize = self.inputImageSize
        self.NumberConvBlocks = NumberConvBlocks
        # while self.NumberConvBlocks > 0:
        #     self.inputImageSize = self.outputImageSize
        #     self.outputImageSize = Calculate_ImageSize_AfterConvBlocks(input_size=self.inputImageSize)
        #     self.NumberConvBlocks = self.NumberConvBlocks - 1
        # self.classifier = nn.Sequential(
        #     nn.Linear(4 * int(np.ceil(3 / 2)) * self.outputImageSize * self.outputImageSize, num_classes),
        # )
        input_s = self.inputImageSize
        featureSize = 0
        for ks in kernel_size_layer:
            featureSize = Calculate_ImageSize_AfterConvBlocks(input_size=input_s, kernel_size=ks)
            input_s = featureSize
        self.classifier = nn.Sequential(nn.Linear(featureSize * featureSize * 12, num_classes))
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, VConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def Student(pretrained=False, **kwargs):
    """Teacher model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # np.random.seed(0)
    #这里的已经证明过，没有用
    # torch.manual_seed(0)
    # torch.cuda.manual_seed(0)
    #randomSeed()
    if pretrained:
        kwargs['init_weights'] = False
    # kwargs['init_weights'] = True
    #model = Student_net5(make_layers(cfg['F'], batch_norm=True), **kwargs)
    # model = Student_net6(make_layersO(cfg['F'], batch_norm=True, ks=kernel_size_layer), **kwargs)
    # model = Student_net2(make_layersO(cfg['B'], batch_norm=True, ks=kernel_size_layer), **kwargs)
    # model = Student_net8(make_layersO(cfg['G'], batch_norm=True, ks=kernel_size_layer), **kwargs)
    model = Student_net9(make_layersO(cfg['H'], batch_norm=True, ks=kernel_size_layer), **kwargs)
    if pretrained:
        # student_model = torch.load('./models' + '/Trained KD student6_Optim model T2 A02 on s115')
        # student_model = torch.load('./models' + '/Trained KD student8_Optim modelK5 T100 A06 on datasetp2p1')
        # student_model = torch.load('./models' + '/Trained KD student8_Optim modelK5 T2 A09 on dataset24')
        # student_model = torch.load('./models' + '/Trained student8 model with KD in data2 Cross scenario T10 A05')
        # student_model = torch.load('./models' + '/Trained student8 model with KD in data3 Cross person T100 A09')
        # student_model = torch.load('./models' + '/Trained student9 model with KD in data2 Cross scenario T75 A09')
        student_model = torch.load('./models' + '/Trained student9 model with KD in data3 Cross person T20 A08')
        # student_model = torch.load('./models' + '/Trained student9 model without Infor in data3 Cross person')
        # student_model = torch.load('./models' + '/Trained student9 model without Infor in data2 Cross scenario')

        # student_model = torch.load('./models' + '/Trained KD student2_Optim model T125 A09 on p2')
        params = student_model.state_dict()  # 获得模型的原始状态以及参数。
        #print(params['features.0.weight'])
        # model_dic = model.state_dict()
        # for k, v in params.items():
        #     # print(k)
        #     # #print(v.shape[0])
        #     # print(model_dic[k].shape)
        #     if not v.shape:
        #         model_dic[k] = v
        #     #print(model_dic[k].shape[0])
        #     elif len(model_dic[k].shape) == 4:
        #         model_dic[k] = v[:model_dic[k].shape[0], :, :, :]
        #     elif len(model_dic[k].shape) == 2:
        #         model_dic[k] = v[:model_dic[k].shape[0], :]
        #     else:
        #         model_dic[k] = v[:model_dic[k].shape[0]]
        # model.load_state_dict(model_dic)
        model.load_state_dict(params)
    return model