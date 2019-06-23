#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import csv


def read_csv(csv_path):
    """

    :param csv_path:
    :return:
    """

    csvFile = open(csv_path, 'r')
    reader = csv.reader(csvFile)
    result = []
    for item in reader:
        result.append(item)
    csvFile.close()
    result = np.array(result)
    return result


def draw(list, save_path):
    plt.style.use("ggplot")
    plt.rcParams['axes.labelweight'] = 'bold'

    x_num = len(list)
    y_min = -0.1
    y_max = 1
    width = 0.5

    # fgsm adversary examples prediction on mutants
    # fgsm_result = read_csv("/Users/krogq/RNNMutaion/result/fgsm_incorrect_final_result_update_r0.05_sd0.1.csv")
    # x_num = int(len(fgsm_result) / 2)
    x_ax = []
    for i in range(x_num):
        x_ax.append(i)
    data2_distance_1 = list
    data2_dis_1 = pd.DataFrame({'x': x_ax, 'diff': data2_distance_1})

    plt.figure(figsize=(6, 5.5))

    plt.plot('x', 'diff', 'x', data=data2_dis_1, color='red', markersize=5)

    plt.xlim(0, x_num)
    plt.ylim(y_min, y_max)
    plt.xlabel('time step', fontsize=15)
    plt.ylabel('difference', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.legend(fontsize=15)
    plt.savefig(save_path + "KS2.pdf")
