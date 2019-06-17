import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.style.use("ggplot")
data1 = np.load("result/lenet5-gf-0.01.npz")
num1 = data1["kill_num"]

data3 = np.load("result/lenet5-gf-0.03.npz")
num3 = data3["kill_num"]

data5 = np.load("result/lenet5-gf-0.05.npz")
num5 = data5["kill_num"]

# y = [len(np.where(num1 > 0)[0]), len(np.where(num3 > 0)[0]), len(np.where(num5 > 0)[0])]
# name_list = ['0.01', '0.03', '0.05']
# x_num = 3
# x_ax = []
# for i in range(x_num):
#     x_ax.append(i)
#
# rects = plt.bar(x_ax, y, color='g')
# plt.xlim(1, x_num)
# plt.ylim(200, 600)
# plt.xlabel('time step', fontsize=15)
# plt.ylabel('difference', fontsize=15)
# plt.xticks(x_ax, name_list)
# plt.yticks(fontsize=15)
# plt.legend(fontsize=12)
# for rect in rects:
#     height = rect.get_height()
#     plt.text(rect.get_x() + rect.get_width() / 2, height, str(height), ha='center', va='bottom')
# plt.show()


import matplotlib.pyplot as plt
import matplotlib
# 设置中文字体和负号正常显示
plt.rcParams['axes.labelweight'] = 'bold'

label_list = ['1-50', '51-100', '101-150', '151-200']    # 横坐标刻度显示值
print(len(np.where((1 < num1) & (num1 < 51))[0]))
num_list1 = [len(np.where((1 < num1) & (num1 < 51))[0]), len(np.where((50 < num1) & (num1 < 101))[0]), len(np.where((100 < num1) & (num1 < 151))[0])]      # 纵坐标值1
num_list2 = [len(np.where((1 < num3) & (num3 < 51))[0]), len(np.where((50 < num3) & (num3< 101))[0]), len(np.where((100 < num3) & (num3 < 151))[0])]       # 纵坐标值2
num_list3 = [len(np.where((1 < num5) & (num5 < 51))[0]), len(np.where((50 < num5) & (num5< 101))[0]), len(np.where((100 < num5) & (num5 < 151))[0])]       # 纵坐标值1
x = range(len(num_list1))
"""
绘制条形图
left:长条形中点横坐标
height:长条形高度
width:长条形宽度，默认值0.8
label:为后面设置legend准备
"""
rects1 = plt.bar(x, num_list1, width=0.3, alpha=0.8, color='red', label="0.01")
rects2 = plt.bar([i + 0.3 for i in x], num_list2, width=0.3, color='green', label="0.03")
rects3 = plt.bar([i + 0.6 for i in x], num_list3, width=0.3, color='blue', label="0.05")
plt.ylim(0, 300)     # y轴取值范围
plt.ylabel("number", fontsize=15)
"""
设置x轴刻度显示值
参数一：中点坐标
参数二：显示值
"""
plt.xticks([index + 0.2 for index in x], label_list)
plt.xlabel("killed number range", fontsize=15)
# plt.title("某某公司")
plt.legend()     # 设置题注
# 编辑文本
for rect in rects1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), ha="center", va="bottom")
for rect in rects2:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), ha="center", va="bottom")
for rect in rects3:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height + 1, str(height), ha="center", va="bottom")

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.legend(fontsize=15)
plt.savefig("mutants_kill.pdf")
plt.show()


