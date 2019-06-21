import time
from keras.datasets import mnist, cifar10
import numpy as np
from cnn_mutation.src.utils import summary_model

# data = np.load("data/select_data.npz")
# indices = data["index"]
# num = data["kill_num"]

data = np.load("data/select_data.npy")
# x = data['x_select']
# x = x.reshape(1, 80)
# np.save("data/select_data.npy", x)
print(data.shape)
# print(indices[-10:])
# print(num[-10:])
# print(np.sum(num)/len(num))
# print(len(np.where(num > 0)[0]))

# from keras.models import load_model
# model = load_model("models/imdb_lstm.h5")
# weight_count, neuron_count, weights_dict, neuron_dict = summary_model(model)
# model.summary()
# print("neuron_count: ", neuron_count)












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
