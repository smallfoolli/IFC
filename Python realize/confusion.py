import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

def cm_plot(original_label, predict_label):
    #print(np.arange(121))
    #cm = confusion_matrix(original_label, predict_label, labels=np.arange(-180, 179, 5))  # 由原标签和预测标签生成混淆矩阵
    cm = confusion_matrix(original_label, predict_label)   # 由原标签和预测标签生成混淆矩阵
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 这里也是一个二维矩阵
    #tick_marks = np.array(range(len(original_label))) + 0.5
    #plt.figure()
    plt.matshow(cm_normalized, cmap=plt.cm.Blues)     # 画混淆矩阵，配色风格使用cm.Blues
    plt.colorbar()    # 颜色标签
    # 显示数据
    #https://blog.csdn.net/qq_36982160/article/details/80038380

    ind_array = np.arange(len(cm_normalized))
    x, y = np.meshgrid(ind_array, ind_array)
    #q = []
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_normalized[y_val][x_val]
        if (c > 0.001) and (c < 0.3):
            plt.text(x_val, y_val, '%0.2f' % (c,), color='k', va='center', ha='center')
        elif c >= 0.3:
            plt.text(x_val, y_val, '%0.2f' % (c,), color='w', va='center', ha='center')


    #plt.xlim(0, 72)
    #plt.ylim(72, 0)
    # https: // www.jb51.net / article / 206130.htm
    plt.ylabel('True label')  # 坐标轴标签
    plt.xlabel('Predicted label')  # 坐标轴标签
    #plt.title('confusion matrix')
    #plt.gca().xaxis.set_ticks_position('bottom')

    #plt.show()
