import pandas as pd
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser(description='Hyperparameter')
parser.add_argument('--Epoch', type=int, default=50, help='The epoch')
parser.add_argument('--Result_save_path', type=str, default='./results/', help='The result save path')
args = parser.parse_args()

#开始读数据
teacher_data = pd.read_csv(args.Result_save_path + 'Teacher results on datasetp2.csv')
student_without_KD_data = pd.read_csv(args.Result_save_path + 'Student2 without KD results on datasetp2.csv')
student_with_KD_data = pd.read_csv(args.Result_save_path + 'Student2 with KD results T20 A06 on datasetp2.csv')
student_with_KD_data_ = pd.read_csv(args.Result_save_path + 'Student2 with KD results T1 A06 on datasetp2.csv')
#student_with_KD_data_ = pd.read_csv(args.Result_save_path + 'Student2 with KD results T1 A06 on datasetp2.csv')
#开始画图
x = list(range(1, args.Epoch+1))

#plt.subplot(2, 1, 1)
plt.figure(1)
plt.plot(x, teacher_data['Test Accuracy']*100, 'b', label='teacher')
plt.plot(x, student_without_KD_data['Test Accuracy']*100, 'k', label='student without KD')
plt.plot(x, student_with_KD_data['Test Accuracy']*100, 'r', label='student with KD')
plt.plot(x, student_with_KD_data_['Test Accuracy']*100, 'r', label='student with KD no T')
plt.xlabel('Epoch number')
plt.ylabel('Accuracy(%)')
#plt.title('Test Accuracy')
plt.legend()

#https://blog.51cto.com/poseidon2011/1900596

#plt.subplot(2, 1, 2)
plt.figure(2)
plt.plot(x, teacher_data['Test Loss'], 'b', label='teacher')
plt.plot(x, student_without_KD_data['Test Loss'], 'k', label='student without KD')
plt.plot(x, student_with_KD_data['Test Loss'], 'r', label='student with KD')
plt.plot(x, student_with_KD_data_['Test Loss'], 'r', label='student with KD no T')
plt.xlabel('Epoch number')
plt.ylabel('Loss')
#plt.title('Test loss')
plt.legend()
plt.show()
