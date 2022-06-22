import torch
import torch.nn as nn
import math
torch.manual_seed(0)
def Calculate_ImageSize_AfterConvBlocks(input_size, kernel_size=3, stride=1, padding=1):
    conv_size = (input_size - kernel_size + 2 * padding) // stride + 1
    after_pool = math.ceil((conv_size - 2) / 2) + 1
    return after_pool
kernel_size_layer = [7, 7]

class StudentNet(nn.Module):
    def __init__(self, inputImageSize, outputImageSize=0, NumberConvBlocks=2):
        super(StudentNet, self).__init__()
        self.inputImageSize = inputImageSize
        self.outputImageSize = outputImageSize
        self.outputImageSize = self.inputImageSize
        self.NumberConvBlocks = NumberConvBlocks
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  ##输入(1, 96, 96, 3)  (3, 1, 3, 64)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 25, kernel_size=3, stride=1, padding=1),##输入(1, 48, 48, 64)  (3, 1, 64, 128)
            nn.BatchNorm2d(25),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            #nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),##输入(1, 24, 24, 128)  (3, 1, 128, 256)
           #nn.BatchNorm2d(32),
            #nn.ReLU(inplace=True),
            #nn.MaxPool2d(2, 2)

            #nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), ##输入(1, 12, 12, 256)  (3, 1, 256, 512)
            #nn.BatchNorm2d(64),
            #nn.ReLU(inplace=True),
            #nn.MaxPool2d(2, 2)

        )

        while self.NumberConvBlocks > 0:
            self.inputImageSize = self.outputImageSize
            self.outputImageSize = Calculate_ImageSize_AfterConvBlocks(input_size=self.inputImageSize)
            self.NumberConvBlocks = self.NumberConvBlocks - 1

        self.classifier = nn.Sequential(nn.Linear(self.outputImageSize * self.outputImageSize * 25, 8))

    def forward(self, x):
        out = self.features(x)
        out = out.reshape(out.size(0), -1)
        out = self.classifier(out)
        return out


class StudentNet1(nn.Module):
    def __init__(self, inputImageSize, outputImageSize=0, NumberConvBlocks=2):
        super(StudentNet1, self).__init__()
        self.inputImageSize = inputImageSize
        self.outputImageSize = outputImageSize
        self.outputImageSize = self.inputImageSize
        self.NumberConvBlocks = NumberConvBlocks
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  ##输入(1, 96, 96, 3)  (3, 1, 3, 64)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),##输入(1, 48, 48, 64)  (3, 1, 64, 128)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            #nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),##输入(1, 24, 24, 128)  (3, 1, 128, 256)
           #nn.BatchNorm2d(32),
            #nn.ReLU(inplace=True),
            #nn.MaxPool2d(2, 2)

            #nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), ##输入(1, 12, 12, 256)  (3, 1, 256, 512)
            #nn.BatchNorm2d(64),
            #nn.ReLU(inplace=True),
            #nn.MaxPool2d(2, 2)

        )

        while self.NumberConvBlocks > 0:
            self.inputImageSize = self.outputImageSize
            self.outputImageSize = Calculate_ImageSize_AfterConvBlocks(input_size=self.inputImageSize)
            self.NumberConvBlocks = self.NumberConvBlocks - 1

        self.classifier = nn.Sequential(nn.Linear(self.outputImageSize * self.outputImageSize * 16, 8))

    def forward(self, x):
        out = self.features(x)
        out = out.reshape(out.size(0), -1)
        out = self.classifier(out)
        return out


class StudentNet2(nn.Module):
    def __init__(self, inputImageSize, outputImageSize=0, NumberConvBlocks=2):
        super(StudentNet2, self).__init__()
        self.inputImageSize = inputImageSize
        self.outputImageSize = outputImageSize
        self.outputImageSize = self.inputImageSize
        self.NumberConvBlocks = NumberConvBlocks
        self.features = nn.Sequential(
            nn.Conv2d(3, 9, kernel_size=3, stride=1, padding=1),  ##输入(1, 96, 96, 3)  (3, 1, 3, 64)
            nn.BatchNorm2d(9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(9, 9, kernel_size=3, stride=1, padding=1),##输入(1, 48, 48, 64)  (3, 1, 64, 128)
            nn.BatchNorm2d(9),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            #nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),##输入(1, 24, 24, 128)  (3, 1, 128, 256)
           #nn.BatchNorm2d(32),
            #nn.ReLU(inplace=True),
            #nn.MaxPool2d(2, 2)

            #nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), ##输入(1, 12, 12, 256)  (3, 1, 256, 512)
            #nn.BatchNorm2d(64),
            #nn.ReLU(inplace=True),
            #nn.MaxPool2d(2, 2)

        )

        while self.NumberConvBlocks > 0:
            self.inputImageSize = self.outputImageSize
            self.outputImageSize = Calculate_ImageSize_AfterConvBlocks(input_size=self.inputImageSize)
            self.NumberConvBlocks = self.NumberConvBlocks - 1

        self.classifier = nn.Sequential(nn.Linear(self.outputImageSize * self.outputImageSize * 9, 10))

    def forward(self, x):
        out = self.features(x)
        out = out.reshape(out.size(0), -1)
        out = self.classifier(out)
        return out


class StudentNet3(nn.Module):
    def __init__(self, inputImageSize, outputImageSize=0, NumberConvBlocks=2):
        super(StudentNet3, self).__init__()
        self.inputImageSize = inputImageSize
        self.outputImageSize = outputImageSize
        self.outputImageSize = self.inputImageSize
        self.NumberConvBlocks = NumberConvBlocks
        self.features = nn.Sequential(
            nn.Conv2d(3, 2, kernel_size=3, stride=1, padding=1),  ##输入(1, 96, 96, 3)  (3, 1, 3, 64)
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1),##输入(1, 48, 48, 64)  (3, 1, 64, 128)
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            #nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),##输入(1, 24, 24, 128)  (3, 1, 128, 256)
           #nn.BatchNorm2d(32),
            #nn.ReLU(inplace=True),
            #nn.MaxPool2d(2, 2)

            #nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), ##输入(1, 12, 12, 256)  (3, 1, 256, 512)
            #nn.BatchNorm2d(64),
            #nn.ReLU(inplace=True),
            #nn.MaxPool2d(2, 2)

        )

        while self.NumberConvBlocks > 0:
            self.inputImageSize = self.outputImageSize
            self.outputImageSize = Calculate_ImageSize_AfterConvBlocks(input_size=self.inputImageSize)
            self.NumberConvBlocks = self.NumberConvBlocks - 1

        self.classifier = nn.Sequential(nn.Linear(self.outputImageSize * self.outputImageSize * 2, 10))

    def forward(self, x):
        out = self.features(x)
        out = out.reshape(out.size(0), -1)
        out = self.classifier(out)
        return out

class StudentNet4(nn.Module):
    def __init__(self, inputImageSize, outputImageSize=0, NumberConvBlocks=1):
        super(StudentNet4, self).__init__()
        self.inputImageSize = inputImageSize
        self.outputImageSize = outputImageSize
        self.outputImageSize = self.inputImageSize
        self.NumberConvBlocks = NumberConvBlocks
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),  ##输入(1, 96, 96, 3)  (3, 1, 3, 64)
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            #nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1),##输入(1, 48, 48, 64)  (3, 1, 64, 128)
            #nn.BatchNorm2d(2),
            #nn.ReLU(inplace=True),
            #nn.MaxPool2d(2, 2),

            #nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),##输入(1, 24, 24, 128)  (3, 1, 128, 256)
           #nn.BatchNorm2d(32),
            #nn.ReLU(inplace=True),
            #nn.MaxPool2d(2, 2)

            #nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), ##输入(1, 12, 12, 256)  (3, 1, 256, 512)
            #nn.BatchNorm2d(64),
            #nn.ReLU(inplace=True),
            #nn.MaxPool2d(2, 2)

        )

        while self.NumberConvBlocks > 0:
            self.inputImageSize = self.outputImageSize
            self.outputImageSize = Calculate_ImageSize_AfterConvBlocks(input_size=self.inputImageSize)
            self.NumberConvBlocks = self.NumberConvBlocks - 1

        self.classifier = nn.Sequential(nn.Linear(self.outputImageSize * self.outputImageSize * 8, 8))

    def forward(self, x):
        out = self.features(x)
        out = out.reshape(out.size(0), -1)
        out = self.classifier(out)
        return out

class StudentNet5(nn.Module):
    def __init__(self, inputImageSize, outputImageSize=0, NumberConvBlocks=1):
        super(StudentNet5, self).__init__()
        self.inputImageSize = inputImageSize
        self.outputImageSize = outputImageSize
        self.outputImageSize = self.inputImageSize
        self.NumberConvBlocks = NumberConvBlocks
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1),  ##输入(1, 96, 96, 3)  (3, 1, 3, 64)
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            #nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1),##输入(1, 48, 48, 64)  (3, 1, 64, 128)
            #nn.BatchNorm2d(2),
            #nn.ReLU(inplace=True),
            #nn.MaxPool2d(2, 2),

            #nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),##输入(1, 24, 24, 128)  (3, 1, 128, 256)
           #nn.BatchNorm2d(32),
            #nn.ReLU(inplace=True),
            #nn.MaxPool2d(2, 2)

            #nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), ##输入(1, 12, 12, 256)  (3, 1, 256, 512)
            #nn.BatchNorm2d(64),
            #nn.ReLU(inplace=True),
            #nn.MaxPool2d(2, 2)

        )

        while self.NumberConvBlocks > 0:
            self.inputImageSize = self.outputImageSize
            self.outputImageSize = Calculate_ImageSize_AfterConvBlocks(input_size=self.inputImageSize)
            self.NumberConvBlocks = self.NumberConvBlocks - 1

        self.classifier = nn.Sequential(nn.Linear(self.outputImageSize * self.outputImageSize * 4, 8))

    def forward(self, x):
        out = self.features(x)
        out = out.reshape(out.size(0), -1)
        out = self.classifier(out)
        return out

class StudentNet6(nn.Module):
    def __init__(self, inputImageSize, outputImageSize=0, NumberConvBlocks=2):
        super(StudentNet6, self).__init__()
        self.inputImageSize = inputImageSize
        self.outputImageSize = outputImageSize
        self.outputImageSize = self.inputImageSize
        self.NumberConvBlocks = NumberConvBlocks
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=1),  ##输入(1, 96, 96, 3)  (3, 1, 3, 64)
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, ceil_mode=True),

            nn.Conv2d(6, 4, kernel_size=3, stride=1, padding=1),##输入(1, 48, 48, 64)  (3, 1, 64, 128)
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, ceil_mode=True),

            #nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),##输入(1, 24, 24, 128)  (3, 1, 128, 256)
           #nn.BatchNorm2d(32),
            #nn.ReLU(inplace=True),
            #nn.MaxPool2d(2, 2)

            #nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), ##输入(1, 12, 12, 256)  (3, 1, 256, 512)
            #nn.BatchNorm2d(64),
            #nn.ReLU(inplace=True),
            #nn.MaxPool2d(2, 2)

        )

        # while self.NumberConvBlocks > 0:
        #     self.inputImageSize = self.outputImageSize
        #     self.outputImageSize = Calculate_ImageSize_AfterConvBlocks(input_size=self.inputImageSize)
        #     self.NumberConvBlocks = self.NumberConvBlocks - 1
        input_s = self.inputImageSize
        featureSize = 0
        for ks in kernel_size_layer:
            featureSize = Calculate_ImageSize_AfterConvBlocks(input_size=input_s, kernel_size=ks)
            input_s = featureSize
        self.classifier = nn.Sequential(nn.Linear(featureSize * featureSize * 4, 8))

    def forward(self, x):
        out = self.features(x)
        out = out.reshape(out.size(0), -1)
        out = self.classifier(out)
        return out


class StudentNet7(nn.Module):
    def __init__(self, inputImageSize, outputImageSize=0, NumberConvBlocks=2):
        super(StudentNet7, self).__init__()
        self.inputImageSize = inputImageSize
        self.outputImageSize = outputImageSize
        self.outputImageSize = self.inputImageSize
        self.NumberConvBlocks = NumberConvBlocks
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=1),  ##输入(1, 96, 96, 3)  (3, 1, 3, 64)
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, ceil_mode=True),

            nn.Conv2d(6, 4, kernel_size=3, stride=1, padding=1),##输入(1, 48, 48, 64)  (3, 1, 64, 128)
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, ceil_mode=True),

            #nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),##输入(1, 24, 24, 128)  (3, 1, 128, 256)
           #nn.BatchNorm2d(32),
            #nn.ReLU(inplace=True),
            #nn.MaxPool2d(2, 2)

            #nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), ##输入(1, 12, 12, 256)  (3, 1, 256, 512)
            #nn.BatchNorm2d(64),
            #nn.ReLU(inplace=True),
            #nn.MaxPool2d(2, 2)

        )

        # while self.NumberConvBlocks > 0:
        #     self.inputImageSize = self.outputImageSize
        #     self.outputImageSize = Calculate_ImageSize_AfterConvBlocks(input_size=self.inputImageSize)
        #     self.NumberConvBlocks = self.NumberConvBlocks - 1
        input_s = self.inputImageSize
        featureSize = 0
        for ks in kernel_size_layer:
            featureSize = Calculate_ImageSize_AfterConvBlocks(input_size=input_s, kernel_size=ks)
            input_s = featureSize
        self.classifier = nn.Sequential(nn.Linear(featureSize * featureSize * 4, 10))

    def forward(self, x):
        out = self.features(x)
        out = out.reshape(out.size(0), -1)
        out = self.classifier(out)
        return out

class StudentNet8(nn.Module):
    def __init__(self, inputImageSize, outputImageSize=0, NumberConvBlocks=2):
        super(StudentNet8, self).__init__()
        self.inputImageSize = inputImageSize
        self.outputImageSize = outputImageSize
        self.outputImageSize = self.inputImageSize
        self.NumberConvBlocks = NumberConvBlocks
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1), padding=1),  ##输入(1, 96, 96, 3)  (3, 1, 3, 64)
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, ceil_mode=True),

            # nn.Conv2d(6, 4, kernel_size=3, stride=1, padding=1),##输入(1, 48, 48, 64)  (3, 1, 64, 128)
            # nn.BatchNorm2d(4),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, 2, ceil_mode=True),

            #nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),##输入(1, 24, 24, 128)  (3, 1, 128, 256)
           #nn.BatchNorm2d(32),
            #nn.ReLU(inplace=True),
            #nn.MaxPool2d(2, 2)

            #nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), ##输入(1, 12, 12, 256)  (3, 1, 256, 512)
            #nn.BatchNorm2d(64),
            #nn.ReLU(inplace=True),
            #nn.MaxPool2d(2, 2)

        )

        # while self.NumberConvBlocks > 0:
        #     self.inputImageSize = self.outputImageSize
        #     self.outputImageSize = Calculate_ImageSize_AfterConvBlocks(input_size=self.inputImageSize)
        #     self.NumberConvBlocks = self.NumberConvBlocks - 1
        input_s = self.inputImageSize
        featureSize = 0
        for ks in kernel_size_layer:
            featureSize = Calculate_ImageSize_AfterConvBlocks(input_size=input_s, kernel_size=ks)
            input_s = featureSize
        self.classifier = nn.Sequential(nn.Linear(featureSize * featureSize * 6, 12))

    def forward(self, x):
        out = self.features(x)
        out = out.reshape(out.size(0), -1)
        out = self.classifier(out)
        return out


class StudentNet9(nn.Module):
    def __init__(self, inputImageSize, outputImageSize=0, NumberConvBlocks=2):
        super(StudentNet9, self).__init__()
        self.inputImageSize = inputImageSize
        self.outputImageSize = outputImageSize
        self.outputImageSize = self.inputImageSize
        self.NumberConvBlocks = NumberConvBlocks
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=(7, 7), stride=(1, 1), padding=1),  ##输入(1, 96, 96, 3)  (3, 1, 3, 64)
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, ceil_mode=True),

            nn.Conv2d(6, 12, kernel_size=(7, 7), stride=(1, 1), padding=1),  ##输入(1, 96, 96, 3)  (3, 1, 3, 64)
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, ceil_mode=True),

            # nn.Conv2d(6, 4, kernel_size=3, stride=1, padding=1),##输入(1, 48, 48, 64)  (3, 1, 64, 128)
            # nn.BatchNorm2d(4),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, 2, ceil_mode=True),

            #nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),##输入(1, 24, 24, 128)  (3, 1, 128, 256)
           #nn.BatchNorm2d(32),
            #nn.ReLU(inplace=True),
            #nn.MaxPool2d(2, 2)

            #nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), ##输入(1, 12, 12, 256)  (3, 1, 256, 512)
            #nn.BatchNorm2d(64),
            #nn.ReLU(inplace=True),
            #nn.MaxPool2d(2, 2)

        )

        # while self.NumberConvBlocks > 0:
        #     self.inputImageSize = self.outputImageSize
        #     self.outputImageSize = Calculate_ImageSize_AfterConvBlocks(input_size=self.inputImageSize)
        #     self.NumberConvBlocks = self.NumberConvBlocks - 1
        input_s = self.inputImageSize
        featureSize = 0
        for ks in kernel_size_layer:
            featureSize = Calculate_ImageSize_AfterConvBlocks(input_size=input_s, kernel_size=ks)
            input_s = featureSize
        self.classifier = nn.Sequential(nn.Linear(featureSize * featureSize * 12, 12))

    def forward(self, x):
        out = self.features(x)
        out = out.reshape(out.size(0), -1)
        out = self.classifier(out)
        return out