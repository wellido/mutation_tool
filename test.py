import time
from keras.datasets import mnist, cifar10
import numpy as np

data = np.load("result/test.npz")
indices = data["index"]
num = data["kill_num"]

print(indices[-10:])
print(num[-10:])











# from progressbar import *
#
# total = 1000
#
#
# def dosomework():
#     time.sleep(0.01)
#
#
# pbar = ProgressBar().start()
#
# for i in range(1000):
#     pbar.update(int((i / (total - 1)) * 100))
#     dosomework()
#
# pbar.finish()
