from cnn_mutation.src.cnn_operator import *
from rnn_mutation.src.keras.models import load_model
from rnn_mutation.src.lstm_operator import *
from utils import prepare_mnist_data
import numpy as np
from rnn_mutation.src.utils import *

# model_path = "models/mnist_lenet5.h5"
model_path = "models/imdb_lstm.h5"
# model_path = "models/test.h5"
model = load_model(model_path)
model.summary()
# x = np.load('data/imdb_select_data_100.npz', mmap_mode='r')
# for k in x.files:
#     print(k)
data = np.load("data/select_data.npz")["x_select"]
# print(data.shape)
# # data = data.reshape(1, 80)
x_test = list()
x_test.append(data)
ori_result, new_result = lstm_operator(model, "lstm_1", 3, x_test, time_stop_step=79, standard_deviation=1.0)
print("ori result: ", ori_result)
print("new result: ", new_result)

# (x_train, y_train), (x_test, y_test) = data_preprocess("imdb")
# _, acc_mutant = model_QC(model, x_test, y_test, 32)
# print("mutated model acc: ", acc_mutant)
# y_predict = model.predict(x_test.reshape(len(x_test), 80))
# correct_num = 0
# for i in range(len(y_predict)):
#     if abs(y_test[i] - y_predict[i][0]) <= 0.5:
#         correct_num += 1
#
# print("acc: ", float(correct_num) / len(y_predict) * 100)
# print(y_test[0])

# lstm_operator(model, "lstm_1", 8, None, gate_type=2, ratio=0.5, precision_num=1, standard_deviation=1.0)

# _, acc_mutant = model_QC(model, x_test, y_test, 32, threshold=acc_threshold)
# print("mutated model acc: ", acc_mutant)


# (x_train, y_train), (x_test, y_test) = prepare_mnist_data()
#
# score = model.evaluate(x_test, y_test, verbose=0)
# print('ori accuracy:', score[1])
# # cnn_operator(model, 6, 0.01)
# cnn_operator(model, 6, 0.01)
# score = model.evaluate(x_test, y_test, verbose=0)
# # score = new_model.predict(x_test)
# print('mutant accuracy:', score[0])

# indices = np.random.choice(10, 5, replace=False)
# print(indices)
# test = np.where(indices > 5)[0]
# # print(mutated_indices)
# print(len(test))

