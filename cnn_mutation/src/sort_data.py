import glob
from keras.datasets import mnist
from keras.models import load_model
import keras.backend as K
import gc
import numpy as np
import argparse
from progressbar import *
from drawer import *


def sort_data(ori_model_path, mutants_path, x, y, save_path):
    """

    :param ori_model_path:
    :param mutants_path:
    :param x:
    :param y:
    :param save_path:
    :return:
    """
    # print(mutants_path)
    # model_path = mutants_path
    model_path = glob.glob(mutants_path + '/*.h5')
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
    # print("running time: ", elapsed)
    #
    count_list = np.asarray(count_list)
    sorted_list = np.argsort(count_list[correct_index])
    # save as npz file
    np.savez(save_path + "KS1.npz", index=correct_index[sorted_list], kill_num=count_list[correct_index[sorted_list]])
    draw(save_path + "KS1.npz", save_path)
    print("completed.")


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", "-model_path",
                        type=str,
                        help="ori model path")
    parser.add_argument("--mutants_path", "-mutants_path",
                        type=str,
                        help="mutants folder path")
    parser.add_argument("--save_path", "-save_path",
                        type=str,
                        help="npz save path")
    args = parser.parse_args()
    ori_model_path = args.model_path
    mutants_path = args.mutants_path
    save_path = args.save_path

    ############# modify here for your data preprocession ###############
    (_, __,), (x_test, y_test) = mnist.load_data()
    x_test = x_test.astype('float32')
    x_test = x_test / 255.
    x_test = x_test.reshape(len(x_test), 28, 28, 1)
    ######################################################################

    sort_data(ori_model_path, mutants_path, x_test, y_test, save_path)


if __name__ == '__main__':
    run()
    # ori_model_path = "../../models/mnist_lenet5.h5"
    # mutants_path = "../../../lenet5-mutants2"
    # (_, __,), (x_test, y_test) = mnist.load_data()
    # x_test = x_test.astype('float32')
    # x_test = x_test / 255.
    # x_test = x_test.reshape(len(x_test), 28, 28, 1)
    # save_path = "../../result/"
    # sort_data(ori_model_path, mutants_path, x_test, y_test, save_path)

# python sort_data.py -model_path ../../models/mnist_lenet5.h5 -mutants_path ../../../lenet5-mutants1 -save_path ../../result/