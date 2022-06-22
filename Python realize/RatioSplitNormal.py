# https://www.cnblogs.com/bad-robot/p/9734273.html
# https://blog.csdn.net/xuru_0927/article/details/89190408

import os, random, shutil


def moveFile(fileDir, tarDirTrain, tarDirTest):
    pathDir = os.listdir(fileDir)  # 取图片的原始路径
    filenumber = len(pathDir)
    rate = 0.6  # 自定义抽取图片的比例，比方说100张抽10张，那就是0.1
    picknumber = int(filenumber * rate)  # 按照rate比例从文件夹中取一定数量图片
    sample = random.sample(pathDir, picknumber)  # 随机选取picknumber数量的样本图片
    # print(sample)
    # tem = sample[0].split('_')
    # print(tem[0])
    temp = fileDir.split('/')
    # print(temp[-2])
    # for item in sample:
    #     tem = item.split('_')
    #     print(tem[0])
    #剩下的做测试集，求全集与抽到的样本集的差集
    remain = list(set(pathDir) - set(sample))
    # print(remain)
    # print(len(sample) + len(remain))
    # print(tarDirTrain + tem[0])
    if not os.path.exists(tarDirTrain + '/' + temp[-1]):
        os.mkdir(tarDirTrain + '/' + temp[-1])
    if not os.path.exists(tarDirTest + '/' + temp[-1]):
        os.mkdir(tarDirTest + '/' + temp[-1])
    for name in sample:
        # shutil.move(fileDir + '/' + name, tarDirTrain + '/' + name)
        # if 'come'
        # shutil.copy(fileDir + '/' + name, tarDirTrain + tem[0] + '/' + temp[-2] + name)
        shutil.copy(fileDir + '/' + name, tarDirTrain + '/' + temp[-1] + '/' + name)
    for name in remain:
        # shutil.move(fileDir + '/' + name, tarDirTest + '/' + name)
        shutil.copy(fileDir + '/' + name, tarDirTest + '/' + temp[-1] + '/' + name)
    return


if __name__ == '__main__':
    # fileDir = "./source/"  # 源图片文件夹路径
    # tarDir = './result/'  # 移动到新的文件夹路径
    fileDir1 = "./radarDataNew/data1_12"  # 源图片文件夹路径
    # fileDir1 = "./data/p1/"  # 源图片文件夹路径
    # fileDir2 = "./data/p2/"  # 源图片文件夹路径
    tarDir1 = './radarDataNew/data1_Train'  # 移动到新的文件夹路径
    tarDir2 = './radarDataNew/data1_Test'  # 移动到新的文件夹路径
    #读取下面的所有文件夹，挨个进行操作，调用移动的函数
    clses = os.listdir(fileDir1)    #列出该文件下的所有文件名
    print(clses)
    for file in clses:
        fileS = fileDir1 + '/' + file
        fileTrain = tarDir1
        fileTest = tarDir2
        moveFile(fileS, fileTrain, fileTest)
    # #两个for循环可优化一下
    # for file in clses:
    #     fileS = os.listdir(fileDir1 + '/' + file)
    #     print(fileS)
    #     for filesub in fileS:
    #         fileD = fileDir1 + '/' + file + '/' + filesub
    #         fileTrain = tarDir1
    #         fileTest = tarDir2
    #         moveFile(fileD, fileTrain, fileTest)
    #     # moveFile(fileD, fileTrain, fileTest)
    # # clses = os.listdir(fileDir2)  # 列出该文件下的所有文件名
    # # print(clses)
    # # for file in clses:
    # #     fileD = fileDir2 + file
    # #     fileTrain = tarDir1 + file
    # #     fileTest = tarDir2 + file
    # #     moveFile(fileD, fileTrain, fileTest)
    #
    # #moveFile(fileDir1, tarDir1)
    #
    #
    #
    #
    #
    #











