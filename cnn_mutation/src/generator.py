from cnn_operator import *
from keras.models import load_model
import argparse
from utils import summary_model
from termcolor import colored
from progressbar import *


def cnn_mutants_generation(ori_model, operator, ratio, standard_deviation=0.5):
    """

    :param ori_model:
    :param operator:
    :param ratio:
    :param standard_deviation:
    :return:
    """
    if operator < 5:
        cnn_operator(ori_model, operator, ratio, standard_deviation)
    else:
        new_model = cnn_operator(ori_model, operator, ratio, standard_deviation)
        return new_model
    return ori_model


def generator():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        help="ori model path")
    parser.add_argument("--operator", type=int,
                        help="mutator")
    parser.add_argument("--ratio", type=float,
                        help="mutation ratio")
    parser.add_argument("--save_path", type=str,
                        help="mutants save path")
    parser.add_argument("--num", type=int,
                        default=1,
                        help="mutants number")
    parser.add_argument("--standard_deviation", type=float,
                        default=0.5,
                        help="standard deviation for gaussian fuzzing")
    args = parser.parse_args()
    model_path = args.model_path
    operator = args.operator
    ratio = args.ratio
    num = args.num
    save_path = args.save_path
    standard_deviation = args.standard_deviation
    p_bar = ProgressBar().start()
    i = 1
    model = load_model(model_path)
    weight_count, neuron_count, weights_dict, neuron_dict = summary_model(model)
    if operator == 0:
        print("total weights: ", weight_count)
        print("process weights num: ", int(weight_count * ratio) if int(weight_count * ratio) > 0 else 1)
    else:
        print("total neuron: ", neuron_count)
        print("process neuron num: ", int(neuron_count * ratio) if int(neuron_count * ratio) > 0 else 1)
    while i <= num:
        # print("process model num: ", i)
        if i != 1:
            model = load_model(model_path)
        new_model = cnn_mutants_generation(model, operator, ratio, standard_deviation)
        final_path = save_path + "/" + cnn_operator_name(operator) + "_" + str(ratio) + "_" + str(i) + ".h5"
        new_model.save(final_path)
        p_bar.update(int((i / num) * 100))
        i += 1
    p_bar.finish()


if __name__ == '__main__':
    generator()

# python generator.py --model_path ../../models/mnist_lenet5.h5 --operator 0 --ratio 0.01 --save_path ../../mutants --num 2

