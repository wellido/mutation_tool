import glob
from keras.datasets import mnist
from keras.models import load_model
import keras.backend as K
import gc
import numpy as np
from progressbar import *


def sort_data(ori_model_path, mutants_path, x, y, save_path):
    """

    :param ori_model_path:
    :param mutants_path:
    :param x:
    :param y:
    :param save_path:
    :return:
    """
    model_path = mutants_path
    model_path = glob.glob(model_path + '/*.h5')
    count_list = [0 for i in range(len(x))]
    ori_model = load_model(ori_model_path)
    ori_predict = ori_model.predict(x).argmax(axis=-1)
    correct_index = np.where(ori_predict == y)[0]
    p_bar = ProgressBar().start()
    i = 1
    num = 200
    start_time = time.clock()
    for path in model_path:
        model = load_model(path)
        result = model.predict(x).argmax(axis=-1)
        for index in correct_index:
            if result[index] != ori_predict[index]:
                count_list[index] += 1
        K.clear_session()
        del model
        gc.collect()
        p_bar.update(int((i / num) * 100))
        i += 1
    p_bar.finish()
    elapsed = (time.clock() - start_time)
    print("running time: ", elapsed)
    #
    count_list = np.asarray(count_list)
    print(count_list)
    sorted_list = np.argsort(count_list[correct_index])
    # save as npz file
    np.savez(save_path, index=correct_index[sorted_list], kill_num=count_list[correct_index[sorted_list]])


if __name__ == '__main__':
    ori_model_path = "../../models/mnist_lenet5.h5"
    mutants_path = "../../../lenet5-mutants"
    (_, __,), (x_test, y_test) = mnist.load_data()
    x_test = x_test.astype('float32')
    x_test = x_test / 255.
    x_test = x_test.reshape(len(x_test), 28, 28, 1)
    save_path = "../../result/test.npz"
    sort_data(ori_model_path, mutants_path, x_test, y_test, save_path)