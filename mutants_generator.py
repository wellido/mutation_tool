from cnn_mutation.src.cnn_operator import *
from rnn_mutation.src.gru_operator import *
import argparse
from keras.models import load_model
from utils import *


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


def rnn_mutants_generation(model, model_type, layer_name, operator, data, gate_type, ratio, precision_num,
                           standard_deviation=0.5):
    """

    :param model:
    :param model_type:
    :param layer_name:
    :param operator:
    :param data:
    :param gate_type:
    :param ratio:
    :param precision_num:
    :param standard_deviation:
    :return:
    """
    if model_type == 'lstm':
        lstm_operator(model, layer_name, operator, data, gate_type=gate_type, ratio=ratio,
                      precision_num=precision_num, standard_deviation=standard_deviation)
    else:
        gru_operator(model, layer_name, operator, data, gate_type=gate_type, ratio=ratio,
                     precision_num=precision_num, standard_deviation=standard_deviation)


def rnn_dynamic_mutation(model, model_type, layer_name, operator, data, time_stop_step, standard_deviation, precision_num):
    """

    :param model:
    :param model_type:
    :param layer_name:
    :param operator:
    :param data:
    :param time_stop_step:
    :param standard_deviation:
    :param precision_num:
    :return:
    """
    if model_type == 'lstm':
        ori_result, new_result = lstm_operator(model, layer_name, operator, data,
                                               time_stop_step=time_stop_step, standard_deviation=standard_deviation,
                                               precision_num=precision_num)
    else:
        ori_result, new_result = gru_operator(model, layer_name, operator, data,
                                              time_stop_step=time_stop_step, standard_deviation=standard_deviation,
                                              precision_num=precision_num)
    return ori_result, new_result


def model_quantification(model, x, y):
    """

    :param model:
    :param x:
    :param y:
    :return:
    """
    y_predict = model.predict(x)
    y_class = np.argmax(y_predict, axis=1)
    correct = np.sum(y.flatten() == y_class.flatten())
    acc = float(correct) / len(x)
    return acc


def generator():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        help="ori model path")
    parser.add_argument("--model_type", type=str,
                        help="cnn, lstm, or gru")
    parser.add_argument("--operator", type=int,
                        help="mutator")
    parser.add_argument("--ratio", type=float,
                        help="mutation ratio")
    parser.add_argument("--save_path", type=str,
                        help="mutants save path")
    parser.add_argument("--layer_name", type=str,
                        default='lstm',
                        help="lstm or gru")
    parser.add_argument("--data_path", type=str,
                        default="test",
                        help="data path")
    parser.add_argument("--gate_type", type=int,
                        default=0,
                        help="gate type, 0, 1, 2, 3")
    # parser.add_argument("--threshold", type=float,
    #                     default=0.9,
    #                     help="acc threshold")
    parser.add_argument("--num", type=int,
                        default=1,
                        help="mutants number")
    parser.add_argument("--standard_deviation", type=float,
                        default=0.5,
                        help="standard deviation for gaussian fuzzing")
    parser.add_argument("--precision_num", type=int,
                        default=2,
                        help="remain precision")
    args = parser.parse_args()
    model_path = args.model_path
    model_type = args.model_type
    operator = args.operator
    ratio = args.ratio
    layer_name = args.layer_name
    data_path = args.data_path
    gate_type = args.gate_type
    num = args.num
    save_path = args.save_path
    standard_deviation = args.standard_deviation
    precision_num = args.precision_num
    if model_type == 'cnn':
        i = 1
        while i <= num:
            print("process model num: ", i)
            model = load_model(model_path)
            new_model = cnn_mutants_generation(model, operator, ratio, standard_deviation)
            final_path = save_path + "/" + cnn_operator_name(operator) + "_" + str(ratio) + "_" + str(i)
            new_model.save(final_path)
            i += 1
    else:
        i = 1
        x_test = np.load(data_path)
        while i <= num:
            print("process model num: ", i)
            model = load_model(model_path)
            rnn_mutants_generation(model, model_type, layer_name, operator, x_test, gate_type, ratio, precision_num,
                                   standard_deviation)
            if model_type == 'lstm':
                final_path = save_path + "/" + rnn_operator_name(operator) + "_" + return_lstm_gate_name(gate_type) \
                             + "_" + str(ratio) + "_" + str(i)
                model.save(final_path)
            else:
                final_path = save_path + "/" + rnn_operator_name(operator) + "_" + return_gru_gate_name(gate_type) \
                             + "_" + str(ratio) + "_" + str(i)
                model.save(final_path)
            i += 1


if __name__ == '__main__':
    generator()


# python mutants_generator.py --model_path models/mnist_lenet5.h5 --model_type cnn --operator 0 --ratio 0.01 --save_path mutants --num 1
