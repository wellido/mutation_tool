import numpy as np
from keras.datasets import mnist
from keras import utils


def prepare_mnist_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    y_train, y_test = utils.to_categorical(y_train, 10), utils.to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)


def random_select(total_num, select_num, layer_list, layer_dict):
    """

    :param total_num:
    :param select_num:
    :param layer_list:
    :param layer_dict:
    :return:
    """
    indices = np.random.choice(total_num, select_num, replace=False)
    process_num_list = []
    process_num_total = 0
    for i in range(len(layer_list)):
        if i == 0:
            num = len(np.where(indices < layer_dict[layer_list[i]])[0])
            process_num_list.append(num)
            process_num_total += num
        else:
            num = len(np.where(indices < layer_dict[layer_list[i]])[0])
            num -= process_num_total
            process_num_total += num
            process_num_list.append(num)
    return process_num_list


def cnn_operator_name(int_operator):
    if int_operator == 0:
        return 'GF'
    elif int_operator == 1:
        return 'WS'
    elif int_operator == 2:
        return 'NEB'
    elif int_operator == 3:
        return 'NAI'
    elif int_operator == 4:
        return 'NS'
    elif int_operator == 5:
        return 'LR'
    elif int_operator == 6:
        return 'LA'
    else:
        return 'LD'


def rnn_operator_name(int_operator):
    """

        :param int_operator:
        :return:
        """
    if int_operator == 1:
        return "SSC"
    elif int_operator == 2:
        return "SR"
    elif int_operator == 3:
        return "SGF"
    elif int_operator == 4:
        return "SPR"
    elif int_operator == 5:
        return "DGC"
    elif int_operator == 6:
        return "DGGF"
    elif int_operator == 7:
        return "DGPR"
    elif int_operator == 8:
        return "SGGF"
    elif int_operator == 9:
        return "SGPR"
    elif int_operator == 10:
        return "WGF"
    elif int_operator == 11:
        return "WQ"
    else:
        return "WPR"
