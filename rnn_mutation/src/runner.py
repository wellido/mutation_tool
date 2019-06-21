from gru_operator import *
from lstm_operator import *
from utils import *
import csv
from keras.models import load_model
import argparse
from progressbar import *
from termcolor import colored


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
    x_test = select_data
    start_time = time.clock()
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
    elapsed = (time.clock() - start_time)
    print("running time: ", elapsed)
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
            model = load_model(model_path)
            lstm_operator(model, layer_name, mutant_operator, None, gate_type=gate_type,
                          precision_num=precision_num, standard_deviation=standard_deviation)

        elif layer_type == "gru":
            model = load_model(model_path)
            gru_operator(model, layer_name, mutant_operator, None, gate_type=gate_type,
                          precision_num=precision_num, standard_deviation=standard_deviation)

    elif mutant_operator == 10 or mutant_operator == 11 or mutant_operator == 12:
        if layer_type == "lstm":
            i = 1
            p_bar = ProgressBar().start()
            start_time = time.clock()
            while i <= num:
                model = load_model(model_path)
                lstm_operator(model, layer_name, mutant_operator, None, gate_type=gate_type, ratio=ratio,
                              standard_deviation=standard_deviation, precision_num=precision_num)
                full_save_path = save_path + "/" + return_operator_name(mutant_operator) + "_" + \
                                 return_lstm_gate_name(gate_type) + "_" + str(ratio) + "_" + \
                                 str(precision_num) + "_" + str(standard_deviation) + "_" + str(i) + ".h5"
                p_bar.update(int((i / num) * 100))
                model.save(full_save_path)
                i += 1
                K.clear_session()
                del model
                gc.collect()
            p_bar.finish()
            elapsed = (time.clock() - start_time)
            print("running time: ", elapsed)

        elif layer_type == "gru":
            i = 1
            start_time = time.clock()
            p_bar = ProgressBar().start()
            while i <= num:
                model = load_model(model_path)
                gru_operator(model, layer_name, mutant_operator, None, gate_type=gate_type, ratio=ratio,
                             precision_num=precision_num, standard_deviation=standard_deviation)

                full_save_path = save_path + "/gru_" + return_operator_name(mutant_operator) + "_" + \
                                 return_gru_gate_name(gate_type) + "_" + str(ratio) + "_" + \
                                 str(precision_num) + "_" + str(standard_deviation) + "_" + str(num) + ".h5"
                p_bar.update(int((i / num) * 100))
                model.save(full_save_path)
                i += 1
                K.clear_session()
                del model
                gc.collect()
            p_bar.finish()
            elapsed = (time.clock() - start_time)
            print("running time: ", elapsed)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator_type", "-operator_type",
                        type=str,
                        help="static or dynamic")
    parser.add_argument("--model_path", "-model_path",
                        type=str,
                        help="model path")
    parser.add_argument("--save_path", "-save_path",
                        type=str,
                        help="model save path")
    parser.add_argument("--operator", "-operator",
                        type=int,
                        help="operator")
    parser.add_argument("--single_data_path", "-single_data_path",
                        type=str,
                        default=" ",
                        help=".npz file path which save the selected data")
    parser.add_argument("--layer_type", "-layer_type",
                        type=str,
                        help="lstm or gru")
    parser.add_argument("--layer_name", "-layer_name",
                        type=str,
                        help="layer name")
    parser.add_argument("--rnn_cell_index", "-rnn_cell_index",
                        type=int,
                        help="the index of rnn layer out of all rnn layers")
    parser.add_argument("--ratio", "-ratio",
                        type=float,
                        default=0.01,
                        help="mutation ratio")
    parser.add_argument("--gate_type", "-gate_type",
                        type=int,
                        default=0,
                        help="gate type selected")
    parser.add_argument("--precision_num", "-precision_num",
                        type=int,
                        default=0,
                        help="precision number remain")
    parser.add_argument("--standard_deviation", "-standard_deviation",
                        type=float,
                        default=0.0,
                        help="standard deviation")
    parser.add_argument("--time_stop_step", "-time_stop_step",
                        type=int,
                        default=0,
                        help="stop at witch time step")
    parser.add_argument("--time_start_step", "-time_start_step",
                        type=int,
                        default=0,
                        help="re-start at witch time step")
    parser.add_argument("--csv_path", "-csv_path",
                        type=str,
                        help="save the results")
    parser.add_argument("--num", "-num",
                        type=int,
                        help="number of mutant")
    parser.add_argument("--acc_threshold", "-acc_threshold",
                        type=float,
                        default=0.8,
                        help="mutated model acc threshold")

    args = parser.parse_args()
    operator_type = args.operator_type
    model_path = args.model_path
    save_path = args.save_path
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
    print(colored("operator: %s" % return_operator_name(operator), 'blue'))
    if layer_type == "lstm":
        if operator_type == "static":
            static_runner(model_path, save_path, operator, layer_type, layer_name, num, ratio=ratio,
                          acc_threshold=acc_threshold, gate_type=gate_type,
                          precision_num=precision_num, standard_deviation=standard_deviation)

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
            static_runner(model_path, save_path, operator, layer_type, layer_name, num, ratio=ratio,
                          gate_type=gate_type, acc_threshold=acc_threshold,
                          precision_num=precision_num, standard_deviation=standard_deviation)
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
    # python runner.py --operator_type static --model_path ../../models/imdb_lstm.h5 --save_path ../../../lstm-mutants --num 100 --operator 10 --layer_type lstm --layer_name lstm_1 --ratio 0.01 --gate_type 0 --standard_deviation 0.1
    # python runner.py -operator_type dynamic -model_path ../../models/imdb_lstm.h5 -layer_type lstm -layer_name lstm_1 -rnn_cell_index 1 --operator 1 -single_data_path ../../data/select_data.npz -standard_deviation 1.0 -precision_num 1 -time_stop_step 78 -csv_path "../../result/test.csv"
    # python runner.py --operator_type dynamic --model_path ../../models/imdb_lstm.h5 --layer_type lstm --layer_name lstm_1 --rnn_cell_index 1 --operator 2 --single_data_path ../../data/select_data.npz --standard_deviation 1.0 --precision_num 1 --time_start_step 70 --time_stop_step 78 --csv_path "../../result/test.csv"
