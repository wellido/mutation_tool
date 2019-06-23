import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt

import numpy as np


def draw(data_path, save_path):
    plt.style.use("ggplot")
    data1 = np.load(data_path)
    plt.rcParams['axes.labelweight'] = 'bold'
    label_list = ['1-50', '51-100', '101-150', '151-200']
    num1 = data1["kill_num"]
    num_list1 = [len(np.where((1 < num1) & (num1 < 51))[0]), len(np.where((50 < num1) & (num1 < 101))[0]),
                 len(np.where((100 < num1) & (num1 < 151))[0]), len(np.where((151 < num1))[0])]
    x = range(len(num_list1))
    rects1 = plt.bar(x, num_list1, width=0.3, alpha=0.8, color='red', label='kill_num')
    plt.ylim(0, 300)
    plt.ylabel("number", fontsize=15)
    plt.xticks([index for index in x], label_list)
    plt.xlabel("killed number range", fontsize=15)
    plt.legend()
    for rect in rects1:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 1, str(height), ha="center", va="bottom")
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=15)
    plt.savefig(save_path + "KS1.pdf")
    # plt.show()