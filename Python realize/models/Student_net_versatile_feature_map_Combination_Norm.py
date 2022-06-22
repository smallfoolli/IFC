import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import math
import numpy as np

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

  def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, delta=0, g=2):
    # torch.manual_seed(0)
    # torch.cuda.manual_seed(0)
    # randomSeed()
    kernel_size = _pair(kernel_size)
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    super(VConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
        False, _pair(0), groups, bias, 'zeros')

    self.s_num = int(np.ceil(self.kernel_size[0]/2))
    self.delta = delta
    self.g = g

    self.weight = nn.Parameter(torch.Tensor(
                    int(out_channels), in_channels // groups, *kernel_size))

    self.weightFeatureMap = nn.Parameter(torch.Tensor([1/self.s_num]*self.s_num))

    self.reset_parameters()

  def forward(self, x):

    x_list = []
    s_num = self.s_num
    ch_ratio = (1+self.delta/self.g)
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

            x_list.append(F.conv2d(x1, weight1, self.bias, self.stride,
                                   padding1, self.dilation, self.groups))

    self.weightFeatureMap.data = self.weightFeatureMap.data / sum(self.weightFeatureMap.data)

    for i in range(len(self.weightFeatureMap)):
        # print(self.weightFeatureMap.data[i])
        # print(self.weightFeatureMap[i])
        x_list[i] *= self.weightFeatureMap[i]
    x = sum(x_list)

    return x

class Student_net2(nn.Module):
    # torch.manual_seed(0)
    # torch.cuda.manual_seed(0)
    # np.random.seed(0)
    #这里也没用
    def __init__(self, features, num_classes=8, init_weights=True, inputImageSize=96, outputImageSize=0, NumberConvBlocks=1):
        super(Student_net2, self).__init__()
        # torch.manual_seed(0)
        # torch.cuda.manual_seed(0)
        #randomSeed()
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
        self.classifier = nn.Sequential(nn.Linear(featureSize * featureSize * 9, num_classes))
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



class Student_net5(nn.Module):

    def __init__(self, features, num_classes=8, init_weights=True, inputImageSize=96, outputImageSize=0, NumberConvBlocks=1):
        super(Student_net5, self).__init__()
        self.features = features
        self.inputImageSize = inputImageSize
        self.outputImageSize = outputImageSize
        self.outputImageSize = self.inputImageSize
        self.NumberConvBlocks = NumberConvBlocks
        while self.NumberConvBlocks > 0:
            self.inputImageSize = self.outputImageSize
            self.outputImageSize = Calculate_ImageSize_AfterConvBlocks(input_size=self.inputImageSize)
            self.NumberConvBlocks = self.NumberConvBlocks - 1
        self.classifier = nn.Sequential(
            nn.Linear(4 * int(np.ceil(3/2)) * self.outputImageSize * self.outputImageSize, num_classes),
        )
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


class Student_net6(nn.Module):
    def __init__(self, features, num_classes=8, init_weights=True, inputImageSize=96, outputImageSize=0,
                 NumberConvBlocks=1):
        super(Student_net6, self).__init__()
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
        self.classifier = nn.Sequential(nn.Linear(featureSize * featureSize * 4, num_classes))
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


class Student_net7(nn.Module):
    def __init__(self, features, num_classes=10, init_weights=True, inputImageSize=96, outputImageSize=0,
                 NumberConvBlocks=1):
        super(Student_net7, self).__init__()
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
        self.classifier = nn.Sequential(nn.Linear(featureSize * featureSize * 4, num_classes))
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

class Student_net8(nn.Module):
    def __init__(self, features, num_classes=12, init_weights=True, inputImageSize=96, outputImageSize=0,
                 NumberConvBlocks=1):
        super(Student_net8, self).__init__()
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
        self.classifier = nn.Sequential(nn.Linear(featureSize * featureSize * 6, num_classes))
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
        # student_model = torch.load('./models' + '/Trained student9 model with KD in data3 Cross person T20 A08')
        student_model = torch.load('./models' + '/Trained student9 model with KD_LR0006_BS4 in data3 Cross person T20 A08')
        # student_model = torch.load(
        #     './models' + '/Trained student9 model withlr0006 KD in data2 Cross scenario T20 A08')
        # #
        # student_model = torch.load('./models' + '/Trained student9 model without Infor in data3 Cross person')
        # student_model = torch.load('./models' + '/Trained student9 model without Infor in data2 Cross scenario')

        # student_model = torch.load('./models' + '/Trained KD student2_Optim model T125 A09 on p2')
        params = student_model.state_dict()  # 获得模型的原始状态以及参数。
        #print(params['features.0.weight'])
        model_dic = model.state_dict()
        # for k, v in params.items():
        #     print(k)
        #     #print(v.shape[0])
        #     print(model_dic[k].shape)
        #     if not v.shape:
        #         model_dic[k] = v
        #     #print(model_dic[k].shape[0])
        #     elif len(model_dic[k].shape) == 4:
        #         m = model_dic[k].shape[0] // v.shape[0]
        #         start = 0
        #         for indx in range(1, m+1):
        #             model_dic[k][start:start+v.shape[0], :, :, :] = v
        #             start += v.shape[0]
        #     elif len(model_dic[k].shape) == 2:
        #         if 'classifier' in k:
        #             n = model_dic[k].shape[1] // v.shape[1]
        #             start = 0
        #             for indx in range(n):
        #                 model_dic[k][:, start:start + v.shape[1]] = v
        #                 start += v.shape[1]
        #         else:
        #             m = model_dic[k].shape[0] // v.shape[0]
        #             start = 0
        #             for indx in range(1, m + 1):
        #                 model_dic[k][start:start + v.shape[0], :] = v
        #                 start += v.shape[0]
        #         #model_dic[k] = v[:model_dic[k].shape[0], :model_dic[k].shape[1]]
        #     else:
        #         m = model_dic[k].shape[0] // v.shape[0]
        #         start = 0
        #         for indx in range(1, m + 1):
        #             model_dic[k][start:start + v.shape[0]] = v
        #             start += v.shape[0]
        #         #model_dic[k] = v[:model_dic[k].shape[0]]
        #print(model_dic['features.0.weight'])
        #model_dic.update(params)
        #print(model_dic['features.0.weight'])
        for k, v in params.items():
            # print(k)
            # #print(v.shape[0])
            # print(model_dic[k].shape)
            if not v.shape:
                model_dic[k] = v
            #print(model_dic[k].shape[0])
            elif len(model_dic[k].shape) == 4:
                model_dic[k] = v[:model_dic[k].shape[0], :, :, :]
            elif len(model_dic[k].shape) == 2:
                model_dic[k] = v[:model_dic[k].shape[0], :]
            else:
                model_dic[k] = v[:model_dic[k].shape[0]]
        model.load_state_dict(model_dic)
    return model