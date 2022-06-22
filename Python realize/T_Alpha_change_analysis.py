import pandas as pd
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser(description='Hyperparameter')
parser.add_argument('--Result_save_path', type=str, default='./result/', help='The result save path')
parser.add_argument('--Epoch', type=int, default=55, help='The epoch')
parser.add_argument('--Temperature', type=float, default=[2.0, 5.0, 7.0, 10.0, 20.0, 50.0, 75.0, 100.0,
                                                          125.0, 150.0, 200.0, 250.0, 500.0, 500000.0], help='The Temperature')
#parser.add_argument('--Alpha', type=float, default=[0.9], help='The Alpha')
#parser.add_argument('--Temperature', type=float, default=[41.0, 44.0, 47.0, 50.0, 53.0, 56.0, 59.0, 62.0, 65.0], help='The Temperature')
#parser.add_argument('--Temperature', type=float, default=[85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0, 125.0], help='The Temperature')
# parser.add_argument('--Alpha', type=float, default=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0], help='The Alpha')
parser.add_argument('--Alpha', type=float, default=[0.9, 0.8, 0.7, 0.6, 0.5], help='The Alpha')
args = parser.parse_args()


# data = pd.read_csv(args.Result_save_path + 'Student6_versatile with KD results on dataset_s115_0012AVTV_random.csv')
# data = pd.read_csv(args.Result_save_path + 'Student8 with KD results on dataset_p2randomChange_0016AVTV_random.csv')
# data = pd.read_csv(args.Result_save_path + 'Student8 with KD results on dataset_24randomChange_00006AVTV_random.csv')
# data = pd.read_csv(args.Result_save_path + 'Student8 with KD results on data3_cross persom_0001AVTV_random.csv')
# data = pd.read_csv(args.Result_save_path + 'Student9 with KD results on data2_cross scenario_001AVTV_random.csv')
# data = pd.read_csv(args.Result_save_path + 'Student9 with KD results on data3_cross person_001AVTV_random.csv')
# data = pd.read_csv(args.Result_save_path + 'Student8 with KD results on data2_cross scenario_0001AVTV_random.csv')
# data = pd.read_csv(args.Result_save_path + 'Student9 with KD results on data3_cross person_001AVTV_random.csv')
# data = pd.read_csv(args.Result_save_path + 'Student9FV with KD results on data2_cross scenario_001AVTV_random.csv')
# data = pd.read_csv(args.Result_save_path + 'Student9 with FullInformation results on data2_cross scenario_001AVTV_random.csv')
# data = pd.read_csv(args.Result_save_path + 'Student9 with FullInformation results on data3_cross person_0006AVTV_random.csv')
data = pd.read_csv(args.Result_save_path + 'Student9 with FullInformation results on data3_cross person_LRW0001LR0006AVTV_random.csv')

dataValues = data.values
data_describe = {}
#https://www.cnblogs.com/wuxiangli/p/6046800.html
epoch = 0
for temperature in args.Temperature:
    for alpha in args.Alpha:
        key = 'T=' + str(temperature) + ',' + 'α=' + str(alpha)
        data_describe[key] = dataValues[epoch+args.Epoch-20:epoch+args.Epoch, 5]
        epoch = epoch + args.Epoch

df_result = pd.DataFrame(data_describe)

print(df_result)
Statistic_result = df_result.describe()
print(Statistic_result)
# Statistic_result.to_csv(args.Result_save_path + 'Student8 with KD results statistic describe.csv', index=True, header=True, sep=',')

#plt.figure(figsize=(9.5, 6.5))
df_result.boxplot(showfliers=True, grid=True, showmeans=True)     # 这里，pandas自己有处理的过程。
#showfliers为异常点是否显示
plt.xticks(rotation=45)
plt.ylabel('Accuracy')
plt.xlabel('Temperature, Alpha')    # 我们设置横纵坐标的标题。
plt.title('Accuracy distribution under different conditions')
plt.show()
# #index表示行索引，header表示列索引


'''
开始汇总统计
https://www.cnblogs.com/yan-lei/archive/2017/11/10/7816188.html

print(df_result.shape[1])
for i in range(df_result.shape[1]):
    print(df_result.iloc[:, i])
    #tem = df_result[:, i].describe()
columns = df_result.columns
values = df_result.values

#values.discribe()
#Statistic_result = df_result.describe()
画箱线图的参考网址
https://www.jb51.net/article/173621.htm
https://blog.csdn.net/weixin_42267615/article/details/107809314
'''
