import torch
import torch.nn as nn
import math

def Calculate_ImageSize_AfterConvBlocks(input_size, kernel_size=3, stride=1, padding=1):
    conv_size = (input_size - kernel_size + 2 * padding) // stride + 1
    after_pool = math.ceil((conv_size - 2) / 2) + 1
    return after_pool


class TeacherNet1(nn.Module):
    def __init__(self, inputImageSize, outputImageSize=0, NumberConvBlocks=4):
        super(TeacherNet1, self).__init__()
        self.inputImageSize = inputImageSize
        self.outputImageSize = outputImageSize
        self.outputImageSize = self.inputImageSize
        self.NumberConvBlocks = NumberConvBlocks
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  ##输入(1, 96, 96, 3)  (3, 1, 3, 64)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),##输入(1, 48, 48, 64)  (3, 1, 64, 128)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),##输入(1, 24, 24, 128)  (3, 1, 128, 256)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), ##输入(1, 12, 12, 256)  (3, 1, 256, 512)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        while self.NumberConvBlocks > 0:
            self.inputImageSize = self.outputImageSize
            self.outputImageSize = Calculate_ImageSize_AfterConvBlocks(input_size=self.inputImageSize)
            self.NumberConvBlocks = self.NumberConvBlocks - 1

        self.classifier = nn.Sequential(nn.Linear(self.outputImageSize*self.outputImageSize*512, 12))

    def forward(self, x):
        out = self.features(x)
        out = out.reshape(out.size(0), -1)
        out = self.classifier(out)
        return out


class TeacherNet2(nn.Module):
    def __init__(self, inputImageSize, outputImageSize=0, NumberConvBlocks=4):
        super(TeacherNet2, self).__init__()
        self.inputImageSize = inputImageSize
        self.outputImageSize = outputImageSize
        self.outputImageSize = self.inputImageSize
        self.NumberConvBlocks = NumberConvBlocks
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  ##输入(1, 96, 96, 3)  (3, 1, 3, 64)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),##输入(1, 48, 48, 64)  (3, 1, 64, 128)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),##输入(1, 24, 24, 128)  (3, 1, 128, 256)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), ##输入(1, 12, 12, 256)  (3, 1, 256, 512)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        while self.NumberConvBlocks > 0:
            self.inputImageSize = self.outputImageSize
            self.outputImageSize = Calculate_ImageSize_AfterConvBlocks(input_size=self.inputImageSize)
            self.NumberConvBlocks = self.NumberConvBlocks - 1

        self.classifier = nn.Sequential(
            nn.Linear(self.outputImageSize*self.outputImageSize*512, 4),
            nn.Linear(4, 12)
        )

    def forward(self, x):
        out = self.features(x)
        out = out.reshape(out.size(0), -1)
        out = self.classifier(out)
        return out


class TeacherNet3(nn.Module):
    def __init__(self, inputImageSize, outputImageSize=0, NumberConvBlocks=4):
        super(TeacherNet3, self).__init__()
        self.inputImageSize = inputImageSize
        self.outputImageSize = outputImageSize
        self.outputImageSize = self.inputImageSize
        self.NumberConvBlocks = NumberConvBlocks
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),  ##输入(1, 96, 96, 3)  (3, 1, 3, 64)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=1),##输入(1, 48, 48, 64)  (3, 1, 64, 128)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=1),##输入(1, 24, 24, 128)  (3, 1, 128, 256)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=1), ##输入(1, 12, 12, 256)  (3, 1, 256, 512)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        while self.NumberConvBlocks > 0:
            self.inputImageSize = self.outputImageSize
            self.outputImageSize = Calculate_ImageSize_AfterConvBlocks(input_size=self.inputImageSize)
            self.NumberConvBlocks = self.NumberConvBlocks - 1

        self.classifier = nn.Sequential(
            nn.Linear(self.outputImageSize*self.outputImageSize*512, 1024),
            nn.Linear(1024, 128),
            nn.Linear(128, 12)
        )

    def forward(self, x):
        out = self.features(x)
        out = out.reshape(out.size(0), -1)
        out = self.classifier(out)
        return out



class TeacherNet4(nn.Module):
    def __init__(self, inputImageSize, outputImageSize=0, NumberConvBlocks=4):
        super(TeacherNet4, self).__init__()
        self.inputImageSize = inputImageSize
        self.outputImageSize = outputImageSize
        self.outputImageSize = self.inputImageSize
        self.NumberConvBlocks = NumberConvBlocks
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),  ##输入(1, 96, 96, 3)  (3, 1, 3, 64)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=1),##输入(1, 48, 48, 64)  (3, 1, 64, 128)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=1),##输入(1, 24, 24, 128)  (3, 1, 128, 256)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=1), ##输入(1, 12, 12, 256)  (3, 1, 256, 512)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        while self.NumberConvBlocks > 0:
            self.inputImageSize = self.outputImageSize
            self.outputImageSize = Calculate_ImageSize_AfterConvBlocks(input_size=self.inputImageSize)
            self.NumberConvBlocks = self.NumberConvBlocks - 1

        self.classifier = nn.Sequential(
            nn.Linear(self.outputImageSize*self.outputImageSize*512, 1024),
            nn.Linear(1024, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 12)
        )

    def forward(self, x):
        out = self.features(x)
        out = out.reshape(out.size(0), -1)
        out = self.classifier(out)
        return out


class TeacherNet5(nn.Module):
    def __init__(self, inputImageSize, outputImageSize=0, NumberConvBlocks=4):
        super(TeacherNet5, self).__init__()
        self.inputImageSize = inputImageSize
        self.outputImageSize = outputImageSize
        self.outputImageSize = self.inputImageSize
        self.NumberConvBlocks = NumberConvBlocks
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),  ##输入(1, 96, 96, 3)  (3, 1, 3, 64)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=1),##输入(1, 48, 48, 64)  (3, 1, 64, 128)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=1),##输入(1, 24, 24, 128)  (3, 1, 128, 256)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=1), ##输入(1, 12, 12, 256)  (3, 1, 256, 512)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        while self.NumberConvBlocks > 0:
            self.inputImageSize = self.outputImageSize
            self.outputImageSize = Calculate_ImageSize_AfterConvBlocks(input_size=self.inputImageSize)
            self.NumberConvBlocks = self.NumberConvBlocks - 1

        self.classifier = nn.Sequential(
            nn.Linear(self.outputImageSize*self.outputImageSize*512, 1024),
            nn.Linear(1024, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 32),
            nn.Linear(32, 12)
        )

    def forward(self, x):
        out = self.features(x)
        out = out.reshape(out.size(0), -1)
        out = self.classifier(out)
        return out


class TeacherNet6(nn.Module):
    def __init__(self, inputImageSize, outputImageSize=0, NumberConvBlocks=3):
        super(TeacherNet6, self).__init__()
        self.inputImageSize = inputImageSize
        self.outputImageSize = outputImageSize
        self.outputImageSize = self.inputImageSize
        self.NumberConvBlocks = NumberConvBlocks
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),  ##输入(1, 96, 96, 3)  (3, 1, 3, 64)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=1),##输入(1, 48, 48, 64)  (3, 1, 64, 128)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=1),##输入(1, 24, 24, 128)  (3, 1, 128, 256)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        while self.NumberConvBlocks > 0:
            self.inputImageSize = self.outputImageSize
            self.outputImageSize = Calculate_ImageSize_AfterConvBlocks(input_size=self.inputImageSize)
            self.NumberConvBlocks = self.NumberConvBlocks - 1

        self.classifier = nn.Sequential(
            nn.Linear(self.outputImageSize*self.outputImageSize*256, 1024),
            nn.Linear(1024, 128),
            nn.Linear(128, 12)
        )

    def forward(self, x):
        out = self.features(x)
        out = out.reshape(out.size(0), -1)
        out = self.classifier(out)
        return out


class TeacherNet7(nn.Module):
    def __init__(self, inputImageSize, outputImageSize=0, NumberConvBlocks=2):
        super(TeacherNet7, self).__init__()
        self.inputImageSize = inputImageSize
        self.outputImageSize = outputImageSize
        self.outputImageSize = self.inputImageSize
        self.NumberConvBlocks = NumberConvBlocks
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),  ##输入(1, 96, 96, 3)  (3, 1, 3, 64)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=1),##输入(1, 48, 48, 64)  (3, 1, 64, 128)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        while self.NumberConvBlocks > 0:
            self.inputImageSize = self.outputImageSize
            self.outputImageSize = Calculate_ImageSize_AfterConvBlocks(input_size=self.inputImageSize)
            self.NumberConvBlocks = self.NumberConvBlocks - 1

        self.classifier = nn.Sequential(
            nn.Linear(self.outputImageSize*self.outputImageSize*128, 1024),
            nn.Linear(1024, 128),
            nn.Linear(128, 12)
        )

    def forward(self, x):
        out = self.features(x)
        out = out.reshape(out.size(0), -1)
        out = self.classifier(out)
        return out

class TeacherNet8(nn.Module):
    def __init__(self, inputImageSize, outputImageSize=0, NumberConvBlocks=4):
        super(TeacherNet8, self).__init__()
        self.inputImageSize = inputImageSize
        self.outputImageSize = outputImageSize
        self.outputImageSize = self.inputImageSize
        self.NumberConvBlocks = NumberConvBlocks
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(5, 5), stride=(1, 1), padding=1),  ##输入(1, 96, 96, 3)  (3, 1, 3, 64)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=(5, 5), stride=(1, 1), padding=1),##输入(1, 48, 48, 64)  (3, 1, 64, 128)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=(5, 5), stride=(1, 1), padding=1),##输入(1, 24, 24, 128)  (3, 1, 128, 256)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=(5, 5), stride=(1, 1), padding=1), ##输入(1, 12, 12, 256)  (3, 1, 256, 512)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        # while self.NumberConvBlocks > 0:
        #     self.inputImageSize = self.outputImageSize
        #     self.outputImageSize = Calculate_ImageSize_AfterConvBlocks(input_size=self.inputImageSize, kernel_size=5)
        #     self.NumberConvBlocks = self.NumberConvBlocks - 1

        self.classifier = nn.Sequential(
            nn.Linear(4*4*512, 1024),
            nn.Linear(1024, 128),
            nn.Linear(128, 12)
        )

    def forward(self, x):
        out = self.features(x)
        out = out.reshape(out.size(0), -1)
        out = self.classifier(out)
        return out


class TeacherNet9(nn.Module):
    def __init__(self, inputImageSize, outputImageSize=0, NumberConvBlocks=4):
        super(TeacherNet9, self).__init__()
        self.inputImageSize = inputImageSize
        self.outputImageSize = outputImageSize
        self.outputImageSize = self.inputImageSize
        self.NumberConvBlocks = NumberConvBlocks
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1), padding=1),  ##输入(1, 96, 96, 3)  (3, 1, 3, 64)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=(7, 7), stride=(1, 1), padding=1),##输入(1, 48, 48, 64)  (3, 1, 64, 128)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=(7, 7), stride=(1, 1), padding=1),##输入(1, 24, 24, 128)  (3, 1, 128, 256)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=(7, 7), stride=(1, 1), padding=1), ##输入(1, 12, 12, 256)  (3, 1, 256, 512)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        # while self.NumberConvBlocks > 0:
        #     self.inputImageSize = self.outputImageSize
        #     self.outputImageSize = Calculate_ImageSize_AfterConvBlocks(input_size=self.inputImageSize, kernel_size=5)
        #     self.NumberConvBlocks = self.NumberConvBlocks - 1

        self.classifier = nn.Sequential(
            nn.Linear(2 * 2 * 512, 1024),
            nn.Linear(1024, 128),
            nn.Linear(128, 12)
        )

    def forward(self, x):
        out = self.features(x)
        out = out.reshape(out.size(0), -1)
        out = self.classifier(out)
        return out


class TeacherNet10(nn.Module):
    def __init__(self, inputImageSize, outputImageSize=0, NumberConvBlocks=5):
        super(TeacherNet10, self).__init__()
        self.inputImageSize = inputImageSize
        self.outputImageSize = outputImageSize
        self.outputImageSize = self.inputImageSize
        self.NumberConvBlocks = NumberConvBlocks
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),  ##输入(1, 96, 96, 3)  (3, 1, 3, 64)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=1),##输入(1, 48, 48, 64)  (3, 1, 64, 128)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=1),##输入(1, 24, 24, 128)  (3, 1, 128, 256)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=1), ##输入(1, 12, 12, 256)  (3, 1, 256, 512)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 2, kernel_size=(3, 3), stride=(1, 1), padding=1),  ##输入(1, 12, 12, 256)  (3, 1, 256, 512)
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        while self.NumberConvBlocks > 0:
            self.inputImageSize = self.outputImageSize
            self.outputImageSize = Calculate_ImageSize_AfterConvBlocks(input_size=self.inputImageSize)
            self.NumberConvBlocks = self.NumberConvBlocks - 1

        self.classifier = nn.Sequential(
            nn.Linear(self.outputImageSize*self.outputImageSize*2, 1024),
            nn.Linear(1024, 128),
            nn.Linear(128, 12)
        )

    def forward(self, x):
        out = self.features(x)
        out = out.reshape(out.size(0), -1)
        out = self.classifier(out)
        return out


class TeacherNet12(nn.Module):
    def __init__(self, inputImageSize, outputImageSize=0, NumberConvBlocks=3):
        super(TeacherNet12, self).__init__()
        self.inputImageSize = inputImageSize
        self.outputImageSize = outputImageSize
        self.outputImageSize = self.inputImageSize
        self.NumberConvBlocks = NumberConvBlocks
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  ##输入(1, 96, 96, 3)  (3, 1, 3, 64)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),##输入(1, 48, 48, 64)  (3, 1, 64, 128)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),##输入(1, 24, 24, 128)  (3, 1, 128, 256)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        while self.NumberConvBlocks > 0:
            self.inputImageSize = self.outputImageSize
            self.outputImageSize = Calculate_ImageSize_AfterConvBlocks(input_size=self.inputImageSize)
            self.NumberConvBlocks = self.NumberConvBlocks - 1

        self.classifier = nn.Sequential(nn.Linear(self.outputImageSize*self.outputImageSize*256, 12))

    def forward(self, x):
        out = self.features(x)
        out = out.reshape(out.size(0), -1)
        out = self.classifier(out)
        return out


class TeacherNet13(nn.Module):
    def __init__(self, inputImageSize, outputImageSize=0, NumberConvBlocks=3):
        super(TeacherNet13, self).__init__()
        self.inputImageSize = inputImageSize
        self.outputImageSize = outputImageSize
        self.outputImageSize = self.inputImageSize
        self.NumberConvBlocks = NumberConvBlocks
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  ##输入(1, 96, 96, 3)  (3, 1, 3, 64)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),##输入(1, 48, 48, 64)  (3, 1, 64, 128)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),##输入(1, 24, 24, 128)  (3, 1, 128, 256)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        while self.NumberConvBlocks > 0:
            self.inputImageSize = self.outputImageSize
            self.outputImageSize = Calculate_ImageSize_AfterConvBlocks(input_size=self.inputImageSize)
            self.NumberConvBlocks = self.NumberConvBlocks - 1

        self.classifier = nn.Sequential(
            nn.Linear(self.outputImageSize * self.outputImageSize * 256, 12),
            nn.Linear(12, 12)
        )

    def forward(self, x):
        out = self.features(x)
        out = out.reshape(out.size(0), -1)
        out = self.classifier(out)
        return out

class TeacherNet14(nn.Module):
    def __init__(self, inputImageSize, outputImageSize=0, NumberConvBlocks=3):
        super(TeacherNet14, self).__init__()
        self.inputImageSize = inputImageSize
        self.outputImageSize = outputImageSize
        self.outputImageSize = self.inputImageSize
        self.NumberConvBlocks = NumberConvBlocks
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  ##输入(1, 96, 96, 3)  (3, 1, 3, 64)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),##输入(1, 48, 48, 64)  (3, 1, 64, 128)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),##输入(1, 24, 24, 128)  (3, 1, 128, 256)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        while self.NumberConvBlocks > 0:
            self.inputImageSize = self.outputImageSize
            self.outputImageSize = Calculate_ImageSize_AfterConvBlocks(input_size=self.inputImageSize)
            self.NumberConvBlocks = self.NumberConvBlocks - 1

        self.classifier = nn.Sequential(
            nn.Linear(self.outputImageSize * self.outputImageSize * 256, 128),
            nn.Linear(128, 12),
            nn.Linear(12, 12)
        )

    def forward(self, x):
        out = self.features(x)
        out = out.reshape(out.size(0), -1)
        out = self.classifier(out)
        return out

class TeacherNet15(nn.Module):
    def __init__(self, inputImageSize, outputImageSize=0, NumberConvBlocks=3):
        super(TeacherNet15, self).__init__()
        self.inputImageSize = inputImageSize
        self.outputImageSize = outputImageSize
        self.outputImageSize = self.inputImageSize
        self.NumberConvBlocks = NumberConvBlocks
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  ##输入(1, 96, 96, 3)  (3, 1, 3, 64)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),##输入(1, 48, 48, 64)  (3, 1, 64, 128)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),##输入(1, 24, 24, 128)  (3, 1, 128, 256)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        while self.NumberConvBlocks > 0:
            self.inputImageSize = self.outputImageSize
            self.outputImageSize = Calculate_ImageSize_AfterConvBlocks(input_size=self.inputImageSize)
            self.NumberConvBlocks = self.NumberConvBlocks - 1

        self.classifier = nn.Sequential(
            nn.Linear(self.outputImageSize * self.outputImageSize * 256, 1024),
            nn.Linear(1024, 128),
            nn.Linear(128, 12),
            nn.Linear(12, 12)
        )

    def forward(self, x):
        out = self.features(x)
        out = out.reshape(out.size(0), -1)
        out = self.classifier(out)
        return out

class TeacherNet16(nn.Module):
    def __init__(self, inputImageSize, outputImageSize=0, NumberConvBlocks=4):
        super(TeacherNet16, self).__init__()
        self.inputImageSize = inputImageSize
        self.outputImageSize = outputImageSize
        self.outputImageSize = self.inputImageSize
        self.NumberConvBlocks = NumberConvBlocks
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  ##输入(1, 96, 96, 3)  (3, 1, 3, 64)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),##输入(1, 48, 48, 64)  (3, 1, 64, 128)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),##输入(1, 24, 24, 128)  (3, 1, 128, 256)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  ##输入(1, 24, 24, 128)  (3, 1, 128, 256)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        while self.NumberConvBlocks > 0:
            self.inputImageSize = self.outputImageSize
            self.outputImageSize = Calculate_ImageSize_AfterConvBlocks(input_size=self.inputImageSize)
            self.NumberConvBlocks = self.NumberConvBlocks - 1

        self.classifier = nn.Sequential(
            nn.Linear(self.outputImageSize * self.outputImageSize * 512, 12)
        )

    def forward(self, x):
        out = self.features(x)
        out = out.reshape(out.size(0), -1)
        out = self.classifier(out)
        return out

class TeacherNet17(nn.Module):
    def __init__(self, inputImageSize, outputImageSize=0, NumberConvBlocks=4):
        super(TeacherNet17, self).__init__()
        self.inputImageSize = inputImageSize
        self.outputImageSize = outputImageSize
        self.outputImageSize = self.inputImageSize
        self.NumberConvBlocks = NumberConvBlocks
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  ##输入(1, 96, 96, 3)  (3, 1, 3, 64)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),##输入(1, 48, 48, 64)  (3, 1, 64, 128)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),##输入(1, 24, 24, 128)  (3, 1, 128, 256)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  ##输入(1, 24, 24, 128)  (3, 1, 128, 256)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        while self.NumberConvBlocks > 0:
            self.inputImageSize = self.outputImageSize
            self.outputImageSize = Calculate_ImageSize_AfterConvBlocks(input_size=self.inputImageSize)
            self.NumberConvBlocks = self.NumberConvBlocks - 1

        self.classifier = nn.Sequential(
            nn.Linear(self.outputImageSize * self.outputImageSize * 512, 5),
            nn.Linear(5, 12)
        )

    def forward(self, x):
        out = self.features(x)
        out = out.reshape(out.size(0), -1)
        out = self.classifier(out)
        return out


class TeacherNet18(nn.Module):
    def __init__(self, inputImageSize, outputImageSize=0, NumberConvBlocks=4):
        super(TeacherNet18, self).__init__()
        self.inputImageSize = inputImageSize
        self.outputImageSize = outputImageSize
        self.outputImageSize = self.inputImageSize
        self.NumberConvBlocks = NumberConvBlocks
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  ##输入(1, 96, 96, 3)  (3, 1, 3, 64)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),##输入(1, 48, 48, 64)  (3, 1, 64, 128)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),##输入(1, 24, 24, 128)  (3, 1, 128, 256)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  ##输入(1, 24, 24, 128)  (3, 1, 128, 256)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        while self.NumberConvBlocks > 0:
            self.inputImageSize = self.outputImageSize
            self.outputImageSize = Calculate_ImageSize_AfterConvBlocks(input_size=self.inputImageSize)
            self.NumberConvBlocks = self.NumberConvBlocks - 1

        self.classifier = nn.Sequential(
            nn.Linear(self.outputImageSize * self.outputImageSize * 512, 1024),
            nn.Linear(1024, 128),
            nn.Linear(128, 32),
            nn.Linear(32, 12)
        )

    def forward(self, x):
        out = self.features(x)
        out = out.reshape(out.size(0), -1)
        out = self.classifier(out)
        return out


class TeacherNet19(nn.Module):
    def __init__(self, inputImageSize, outputImageSize=0, NumberConvBlocks=5):
        super(TeacherNet19, self).__init__()
        self.inputImageSize = inputImageSize
        self.outputImageSize = outputImageSize
        self.outputImageSize = self.inputImageSize
        self.NumberConvBlocks = NumberConvBlocks
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  ##输入(1, 96, 96, 3)  (3, 1, 3, 64)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),##输入(1, 48, 48, 64)  (3, 1, 64, 128)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),##输入(1, 24, 24, 128)  (3, 1, 128, 256)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  ##输入(1, 24, 24, 128)  (3, 1, 128, 256)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1),  ##输入(1, 24, 24, 128)  (3, 1, 128, 256)
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        while self.NumberConvBlocks > 0:
            self.inputImageSize = self.outputImageSize
            self.outputImageSize = Calculate_ImageSize_AfterConvBlocks(input_size=self.inputImageSize)
            self.NumberConvBlocks = self.NumberConvBlocks - 1

        self.classifier = nn.Sequential(
            nn.Linear(self.outputImageSize * self.outputImageSize * 4, 12),
            nn.Linear(12, 12)
        )

    def forward(self, x):
        out = self.features(x)
        out = out.reshape(out.size(0), -1)
        out = self.classifier(out)
        return out
