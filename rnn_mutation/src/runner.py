from .gru_operator import *
from .lstm_operator import *
from .utils import *
import csv
from keras.models import load_model
import argparse


def dynamic_runner(model_path, layer_name, rnn_cell_index, operation, data_path, layer_type,
                   ratio=0.01, gate_type=0, time_stop_step=0,
                   time_start_step=0, batch_size=0, stateful=False, precision_num=2, standard_deviation=0.0):
    """

    :param model_path:
    :param layer_name:
    :param rnn_cell_index
    :param operation:
    :param data_path:
    :param layer_type:
    :param ratio:
    :param gate_type:
    :param time_stop_step:
    :param time_start_step:
    :param batch_size:
    :param stateful:
    :param precision_num:
    :param standard_deviation:
    :return:
    """

    model = load_model(model_path)
    select_data = np.load(data_path)
    x_test = list()
    x_test.append(select_data["x_select"])
    if len(select_data) == 3:
        x_test.append(select_data["xq_select"])
    y_test = select_data["y_select"]
    if layer_type == "lstm":
        result = lstm_operator(model, layer_name, operation, x_test, rnn_cell_index=rnn_cell_index, time_stop_step=time_stop_step,
                               gate_type=gate_type, ratio=ratio, time_start_step=time_start_step, batch_size=batch_size,
                               stateful=stateful,
                               precision_num=precision_num, standard_deviation=standard_deviation)
    elif layer_type == "gru":
        result = gru_operator(model, layer_name, operation, x_test, time_stop_step=time_stop_step,
                              gate_type=gate_type, ratio=ratio, time_start_step=time_start_step, batch_size=batch_size,
                              stateful=stateful,
                              precision_num=precision_num, standard_deviation=standard_deviation)
    return result


def static_runner(model_path, save_path, mutant_operator, layer_type, layer_name, num,
                  acc_threshold=0.8, ratio=0.01, gate_type=0, precision_num=2, standard_deviation=0.0):
    """

    :param model_path:
    :param save_path:
    :param mutant_operator:
    :param layer_type:
    :param layer_name:
    :param num:
    :param acc_threshold:
    :param ratio:
    :param gate_type:
    :param precision_num:
    :param standard_deviation:
    :return:
    """

    if mutant_operator == 8 or mutant_operator == 9:
        if layer_type == "lstm":
            acc_mutant = 0
            while acc_mutant < acc_threshold:
                model = load_model(model_path)
                lstm_operator(model, layer_name, mutant_operator, None, gate_type=gate_type,
                              precision_num=precision_num, standard_deviation=standard_deviation)
                (x_train, y_train), (x_test, y_test) = data_preprocess("imdb")
                _, acc_mutant = model_QC(model, x_test, y_test, 32)
                print("threshold: ", acc_threshold)
                print("mutated model acc: ", acc_mutant)

            full_save_path = save_path + "/" + return_operator_name(mutant_operator) + "_" + \
                             return_lstm_gate_name(gate_type) + "_" + str(precision_num) + "_" + \
                             str(standard_deviation) + "_" + str(num) + ".h5"
        elif layer_type == "gru":
            acc_mutant = 0
            while acc_mutant < acc_threshold:
                model = load_model(model_path)
                gru_operator(model, layer_name, mutant_operator, None, gate_type=gate_type,
                              precision_num=precision_num, standard_deviation=standard_deviation)
                (x_train, y_train), (x_test, y_test) = data_preprocess("imdb")
                _, acc_mutant = model_QC(model, x_test, y_test, 32, threshold=acc_threshold)
                print("threshold: ", acc_threshold)
                print("mutated model acc: ", acc_mutant)

            full_save_path = save_path + "/" + return_operator_name(mutant_operator) + "_" + \
                             return_gru_gate_name(gate_type) + "_" + str(precision_num) + "_" + \
                             str(standard_deviation) + "_" + str(num) + ".h5"

    elif mutant_operator == 10 or mutant_operator == 11 or mutant_operator == 12:
        if layer_type == "lstm":
            while 1:
                model = load_model(model_path)
                lstm_operator(model, layer_name, mutant_operator, None, gate_type=gate_type, ratio=ratio,
                              standard_deviation=standard_deviation, precision_num=precision_num)
                (x_train, y_train), (x_test, y_test) = data_preprocess("imdb")
                _, acc_mutant = model_QC(model, x_test, y_test, 32, threshold=acc_threshold)
                print("threshold: ", acc_threshold)
                print("mutated model acc: ", acc_mutant)
                if acc_mutant > acc_threshold:
                    break

            full_save_path = save_path + "/" + return_operator_name(mutant_operator) + "_" + \
                             return_lstm_gate_name(gate_type) + "_" + str(ratio) + "_" + \
                             str(precision_num) + "_" + str(standard_deviation) + "_" + str(num) + ".h5"
        elif layer_type == "gru":
            # while 1:
            model = load_model(model_path)
            gru_operator(model, layer_name, mutant_operator, None, gate_type=gate_type, ratio=ratio,
                         precision_num=precision_num, standard_deviation=standard_deviation)
                # (x_train, y_train), (x_test, y_test) = data_preprocess("imdb")
                # _, acc_mutant = model_QC(model, x_test, y_test, 32, threshold=acc_threshold)
                # print("threshold: ", acc_threshold)
                # print("mutated model acc: ", acc_mutant)
                # if acc_mutant > acc_threshold:
                #     break

            full_save_path = save_path + "/gru_" + return_operator_name(mutant_operator) + "_" + \
                             return_gru_gate_name(gate_type) + "_" + str(ratio) + "_" + \
                             str(precision_num) + "_" + str(standard_deviation) + "_" + str(num) + ".h5"

    model.save(full_save_path)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator_type", type=str,
                        help="static or dynamic")
    parser.add_argument("--model_path", type=str,
                        help="model path")
    parser.add_argument("--save_path", type=str,
                        help="model save path")
    parser.add_argument("--mutants_number", type=int,
                        help="mutants generation number")
    parser.add_argument("--operator", type=int,
                        help="operator")
    parser.add_argument("--single_data_path", type=str, default=" ",
                        help=".npz file path which save the selected data")
    parser.add_argument("--layer_type", type=str,
                        help="lstm or gru")
    parser.add_argument("--layer_name", type=str,
                        help="layer name")
    parser.add_argument("--rnn_cell_index", type=int,
                        help="the index of rnn layer out of all rnn layers")
    parser.add_argument("--ratio", type=float, default=0.01,
                        help="mutation ratio")
    parser.add_argument("--gate_type", type=int, default=0,
                        help="gate type selected")
    parser.add_argument("--precision_num", type=int, default=0,
                        help="precision number remain")
    parser.add_argument("--standard_deviation", type=float, default=0.0,
                        help="standard deviation")
    parser.add_argument("--time_stop_step", type=int, default=0,
                        help="stop at witch time step")
    parser.add_argument("--time_start_step", type=int, default=0,
                        help="re-start at witch time step")
    parser.add_argument("--csv_path", type=str,
                        help="save the results")
    parser.add_argument("--num", type=int,
                        help="number of mutant")
    parser.add_argument("--acc_threshold", type=float, default=0.8,
                        help="mutated model acc threshold")

    args = parser.parse_args()
    operator_type = args.operator_type
    model_path = args.model_path
    save_path = args.save_path
    mutants_number = args.mutants_number
    operator = args.operator
    print("operator number: ", operator)
    single_data_path = args.single_data_path
    layer_type = args.layer_type
    layer_name = args.layer_name
    rnn_cell_index = args.rnn_cell_index
    ratio = args.ratio
    gate_type = args.gate_type
    precision_num = args.precision_num
    time_stop_step = args.time_stop_step
    time_start_step = args.time_start_step
    csv_path = args.csv_path
    num = args.num
    acc_threshold = args.acc_threshold
    standard_deviation = args.standard_deviation
    if layer_type == "lstm":
        if operator_type == "static":
            static_runner(model_path, save_path, operator, layer_type, layer_name, ratio=ratio,
                          acc_threshold=acc_threshold, gate_type=gate_type,
                          precision_num=precision_num, standard_deviation=standard_deviation, num=num)

        else:
            original_result, mutant_result = dynamic_runner(model_path, layer_name, rnn_cell_index, operator,
                                                            single_data_path, layer_type, time_stop_step=time_stop_step,
                                                            time_start_step=time_start_step, gate_type=gate_type,
                                                            precision_num=precision_num,
                                                            standard_deviation=standard_deviation)
            print("original result: ", original_result)
            print("time stop step: ", time_stop_step)
            print("mutant result: ", mutant_result)
            distance_result = distance_calculate(original_result, mutant_result)
            print("total distance: ", distance_result)
            csv_file = open(csv_path, "a")
            try:
                if operator == 1:
                    writer = csv.writer(csv_file)
                    writer.writerow((return_operator_name(operator), time_stop_step, distance_result))
                elif operator == 3:
                    writer = csv.writer(csv_file)
                    writer.writerow((return_operator_name(operator), ratio, standard_deviation,
                                     time_stop_step, distance_result))
                elif operator == 4:
                    writer = csv.writer(csv_file)
                    writer.writerow((return_operator_name(operator), precision_num, time_stop_step, distance_result))
            finally:
                csv_file.close()

    elif layer_type == "gru":
        if operator_type == "static":
            print("######")
            static_runner(model_path, save_path, operator, layer_type, layer_name, ratio=ratio,
                          gate_type=gate_type, acc_threshold=acc_threshold,
                          precision_num=precision_num, standard_deviation=standard_deviation, num=num)
        else:
            original_result, mutant_result = dynamic_runner(model_path, layer_name, rnn_cell_index, operator,
                                                            single_data_path, layer_type, time_stop_step=time_stop_step,
                                                            time_start_step=time_start_step, gate_type=gate_type,
                                                            precision_num=precision_num,
                                                            standard_deviation=standard_deviation)
            print("original result: ", original_result)
            print("time stop step: ", time_stop_step)
            print("mutant result: ", mutant_result)
            distance_result = distance_calculate(original_result, mutant_result)
            print("total distance: ", distance_result)
            csv_file = open(csv_path, "a")
            try:
                if operator == 1:
                    writer = csv.writer(csv_file)
                    writer.writerow((return_operator_name(operator), time_stop_step, distance_result))
                elif operator == 3:
                    writer = csv.writer(csv_file)
                    writer.writerow((return_operator_name(operator), ratio, standard_deviation,
                                     time_stop_step, distance_result))
                elif operator == 4:
                    writer = csv.writer(csv_file)
                    writer.writerow((return_operator_name(operator), precision_num, time_stop_step, distance_result))
            finally:
                csv_file.close()


# ../models/imdb_lstm.h5
if __name__ == '__main__':
    run()

    # python runner.py --operator_type static --model_path /Users/krogq/RNNMutaion/models/imdb_lstm.h5 --save_path /Users/krogq/RNNMutaion/generated_model --mutants_number 100 --operator 10 --layer_type lstm --layer_name lstm_1 --ratio 0.05 --gate_type 2 --standard_deviation 0.1
    # python runner.py --operator_type dynamic --model_path /Users/krogq/RNNMutaion/models/imdb_lstm.h5 --layer_type lstm --layer_name lstm_1 --rnn_cell_index 1 --operator 1 --single_data_path /Users/krogq/RNNMutaion/data/select_data.npz --standard_deviation 1.0 --precision_num 1 --time_stop_step 40 --csv_path "../result/test.csv"
    # python runner.py --operator_type dynamic --model_path /Users/krogq/RNNMutaion/models/imdb_gru.h5 --layer_type gru --layer_name gru_1 --rnn_cell_index 1 --operator 1 --single_data_path /Users/krogq/RNNMutaion/data/select_data.npz --time_stop_step 40 --csv_path "../result/gru_test.csv"
    # python runner.py --operator_type dynamic --model_path /Users/krogq/RNNMutaion/models/babi_rnn_q2_epoch20.h5 --layer_type lstm --layer_name lstm_1 --rnn_cell_index 1 --operator 1 --single_data_path /Users/krogq/RNNMutaion/data/babi_select_data.npz --time_stop_step 40 --csv_path "../result/babi_lstm_test.csv"