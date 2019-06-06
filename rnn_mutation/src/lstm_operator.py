from keras.models import Model, load_model
import numpy as np
import keras.backend as K
# from keras.layers import recurrent
# import recurrent
# import state_save as ss
import sys
# sys.path.append('.')
from keras.layers import recurrent

# from ..src import recurrent
import state_save as ss
# import rnn_mutation.src.state_save as ss
import gc


def lstm_operator(model, layer_name, operation, x_test, rnn_cell_index=1, ratio=0.01, gate_type=0, time_stop_step=0,
                  time_start_step=0, batch_size=0, stateful=False, precision_num=2, standard_deviation=0.0):
    """

    :param model:
    :param layer_name:
    :param operation: 1 - state status clear
                      2 - state reset
                      3 - state gaussian fuzzing
                      4 - state precision reduction
                      5 - dynamic gate clear
                      6 - dynamic gate gaussian fuzzing
                      7 - dynamic gate precision reduction
                      8 - static gate gaussian fuzzing
                      9 - static gate precision reduction
                      10 - weight gaussian fuzzing
                      11 - weight quantization
                      12 - weight precision reduction
    :param x_test:
    :param rnn_cell_index:
    :param ratio:
    :param gate_type: 0 - input
                      1 - forget
                      2 - cell candidate
                      3 - output
    :param time_stop_step
    :param time_start_step
    :param batch_size
    :param stateful
    :param precision_num:
    :param standard_deviation:
    :return:
    """
    if operation == 1:
        if stateful:
            stateful_state_status_clear(model, x_test, layer_name, batch_size)
        else:
            return not_stateful_state_status_process(model, layer_name, rnn_cell_index, x_test, time_stop_step,
                                                     operation, precision_num=precision_num,
                                                     standard_deviation=standard_deviation)
    elif operation == 2:
        if stateful:
            stateful_state_reset(model, x_test, layer_name, batch_size)
        else:
            return not_stateful_state_reset(model, layer_name, rnn_cell_index, x_test, time_stop_step, time_start_step)
    elif operation == 3:
        if stateful:
            stateful_state_gaussian_fuzzing(model, x_test, layer_name, 0, batch_size)
        else:
            return not_stateful_state_status_process(model, layer_name, rnn_cell_index, x_test, time_stop_step,
                                                     operation, precision_num=precision_num,
                                                     standard_deviation=standard_deviation)
    elif operation == 4:
        if stateful:
            stateful_state_precision_reduction(model, x_test, layer_name, precision_num, 0, batch_size)
        else:
            return not_stateful_state_status_process(model, layer_name, rnn_cell_index, x_test, time_stop_step,
                                                     operation, precision_num=precision_num,
                                                     standard_deviation=standard_deviation)
    elif operation == 5:
        if gate_type == 0:
            return dynamic_gate_process(model, layer_name, x_test, time_stop_step, 1,
                                                   standard_deviation=standard_deviation)
        elif gate_type == 1:
            return dynamic_gate_process(model, layer_name, x_test, time_stop_step, 3,
                                                   standard_deviation=standard_deviation)
        elif gate_type == 2:
            return dynamic_gate_process(model, layer_name, x_test, time_stop_step, 5,
                                                   standard_deviation=standard_deviation)
        else:
            return dynamic_gate_process(model, layer_name, x_test, time_stop_step, 7,
                                                   standard_deviation=standard_deviation)
    elif operation == 6:
        if gate_type == 0:
            return dynamic_gate_process(model, layer_name, x_test, time_stop_step, 2,
                                                   standard_deviation=standard_deviation)
        elif gate_type == 1:
            return dynamic_gate_process(model, layer_name, x_test, time_stop_step, 4,
                                                   standard_deviation=standard_deviation)
        elif gate_type == 2:
            return dynamic_gate_process(model, layer_name, x_test, time_stop_step, 6,
                                                   standard_deviation=standard_deviation)
        else:
            return dynamic_gate_process(model, layer_name, x_test, time_stop_step, 8,
                                                   standard_deviation=standard_deviation)
    elif operation == 7:
        if gate_type == 0:
            return dynamic_gate_process(model, layer_name, x_test, time_stop_step, 11,
                                                   standard_deviation=standard_deviation)
        elif gate_type == 1:
            return dynamic_gate_process(model, layer_name, x_test, time_stop_step, 13,
                                                   standard_deviation=standard_deviation)
        elif gate_type == 2:
            return dynamic_gate_process(model, layer_name, x_test, time_stop_step, 15,
                                                   standard_deviation=standard_deviation)
        else:
            return dynamic_gate_process(model, layer_name, x_test, time_stop_step, 17,
                                                   standard_deviation=standard_deviation)
    elif operation == 8:
        if gate_type == 0:
            static_gate_process(model, 2, standard_deviation, precision_num)
        elif gate_type == 1:
            static_gate_process(model, 4, standard_deviation, precision_num)
        elif gate_type == 2:
            static_gate_process(model, 6, standard_deviation, precision_num)
        else:
            static_gate_process(model, 8, standard_deviation, precision_num)
    elif operation == 9:
        if gate_type == 0:
            static_gate_process(model, 11, standard_deviation, precision_num)
        elif gate_type == 1:
            static_gate_process(model, 13, standard_deviation, precision_num)
        elif gate_type == 2:
            static_gate_process(model, 15, standard_deviation, precision_num)
        else:
            static_gate_process(model, 17, standard_deviation, precision_num)
    elif operation == 10:
        weights_process(model, layer_name, gate_type, ratio, weights_gaussian_fuzzing, standard_deviation=standard_deviation)
    elif operation == 11:
        weights_process(model, layer_name, gate_type, ratio, weights_binary_quantization)
    elif operation == 12:
        weights_process(model, layer_name, gate_type, ratio, weights_precision_reduction, precision_num=precision_num)


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


def get_last_state(model, x_test, layer_name, batch_size=0):
    """

    :param model:
    :param x_test:
    :param layer_name:
    :param batch_size:
    :return:
    """
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
    if batch_size > 0:
        state_sequence, state_final, cell_final = intermediate_layer_model.predict(x_test, batch_size=batch_size)
    else:
        state_sequence, state_final, cell_final = intermediate_layer_model.predict(x_test)
    return state_sequence, state_final, cell_final


def return_layer_index(layers, layer_name):
    """

    :param layers:
    :param layer_name:
    :return:
    """
    for i in range(len(layers)):
        if layers[i].name == layer_name:
            return i
    print(layer_name, " is not in the layers.")


def not_stateful_state_reset(model, layer_name, rnn_cell_index, x_test, time_stop_step, time_start_step):
    """

    :param model:
    :param layer_name:
    :param rnn_cell_index:
    :param x_test:
    :param time_stop_step:
    :param time_start_step:
    :return:
    """
    if len(x_test) == 1:
        x_test = x_test[0]
        original_result = model.predict(x_test.reshape(1, 80))
        layer_index = return_layer_index(model.layers, layer_name)
        new_model = Model(inputs=model.input, outputs=model.layers[layer_index].output)
        input_length = len(x_test)
        x_time_stop = x_test[:time_stop_step].reshape(1, time_stop_step)
        x_time_stop_result = new_model.predict(x_time_stop)
        initial_cell = x_time_stop_result[2][0]
        x_time_start = x_test[:time_start_step].reshape(1, time_start_step)
        x_time_start_result = new_model.predict(x_time_start)
        initial_state = x_time_start_result[1][0]
        ss.hidden_state = initial_state
        ss.hidden_cell = initial_cell
        recurrent.global_special_regulation(1)
        recurrent.rnn_cell_index_set(rnn_cell_index)
        model_weights = model.get_weights()
        model_config = model.get_config()
        new_initial_model = Model.from_config(model_config)
        new_initial_model.set_weights(model_weights)
        x_back_part = x_test[time_start_step:].reshape(1, input_length - time_start_step)
        new_result = new_initial_model.predict(x_back_part)
        return original_result, new_result
    elif len(x_test) == 2:
        if rnn_cell_index == 1:
            xq_test = x_test[1]
            x_test = x_test[0]
            original_result = model.predict([x_test.reshape(1, 552), xq_test.reshape(1, 5)])
            layer_index = return_layer_index(model.layers, layer_name)
            new_model = Model(inputs=model.input, outputs=model.layers[layer_index].output)
            input_length = len(x_test)
            x_time_stop = x_test[:time_stop_step].reshape(1, time_stop_step)
            x_time_stop_result = new_model.predict([x_time_stop, xq_test])
            initial_cell = x_time_stop_result[2][0]
            x_time_start = x_test[:time_start_step].reshape(1, time_start_step)
            x_time_start_result = new_model.predict([x_time_start, xq_test])
            initial_state = x_time_start_result[1][0]
            ss.hidden_state = initial_state
            ss.hidden_cell = initial_cell
            recurrent.global_special_regulation(1)
            recurrent.rnn_cell_index_set(rnn_cell_index)
            model_weights = model.get_weights()
            model_config = model.get_config()
            new_initial_model = Model.from_config(model_config)
            new_initial_model.set_weights(model_weights)
            x_back_part = x_test[time_start_step:].reshape(1, input_length - time_start_step)
            new_result = new_initial_model.predict([x_back_part, xq_test])
            return original_result, new_result
        elif rnn_cell_index == 2:
            xq_test = x_test[1]
            x_test = x_test[0]
            original_result = model.predict([x_test.reshape(1, 552), xq_test.reshape(1, 5)])
            layer_index = return_layer_index(model.layers, layer_name)
            new_model = Model(inputs=model.input, outputs=model.layers[layer_index].output)
            input_length = len(xq_test)
            x_time_stop = xq_test[:time_stop_step].reshape(1, time_stop_step)
            x_time_stop_result = new_model.predict([x_test, x_time_stop])
            initial_cell = x_time_stop_result[2][0]
            x_time_start = xq_test[:time_start_step].reshape(1, time_start_step)
            x_time_start_result = new_model.predict([x_test, x_time_start])
            initial_state = x_time_start_result[1][0]
            ss.hidden_state = initial_state
            ss.hidden_cell = initial_cell
            recurrent.global_special_regulation(1)
            recurrent.rnn_cell_index_set(rnn_cell_index)
            model_weights = model.get_weights()
            model_config = model.get_config()
            new_initial_model = Model.from_config(model_config)
            new_initial_model.set_weights(model_weights)
            x_back_part = xq_test[time_start_step:].reshape(1, input_length - time_start_step)
            new_result = new_initial_model.predict([x_test, x_back_part])
            return original_result, new_result


def not_stateful_state_status_process(model, layer_name, rnn_cell_index, x_test, time_stop_step, op,
                                      standard_deviation=1.0, precision_num=2):
    """

    :param model:
    :param layer_name:
    :param rnn_cell_index:
    :param x_test:
    :param time_stop_step:
    :param op: 1 - clear
               3 - gaussian fuzzing
               4 - precision reduction
    :param standard_deviation:
    :param precision_num
    :return:
    """
    if len(x_test) == 1:
        x_test = x_test[0]
        original_result = model.predict(x_test.reshape(1, 80))
        layer_index = return_layer_index(model.layers, layer_name)
        new_model = Model(inputs=model.input, outputs=model.layers[layer_index].output)
        input_length = len(x_test)
        x_front_part = x_test[:time_stop_step].reshape(1, time_stop_step)
        x_back_part = x_test[time_stop_step:].reshape(1, input_length - time_stop_step)
        x_front_result = new_model.predict(x_front_part)
        initial_state = x_front_result[1][0]
        initial_cell = x_front_result[2][0]
        # print("initial_state: ", initial_state)
        # print("initial_cell: ", initial_cell)
        if op == 1:
            initial_cell = initial_cell * 0
        elif op == 3:
            fuzz = np.random.normal(loc=0.0, scale=standard_deviation, size=None)
            print("standard_deviation:", standard_deviation)
            print("fuzz:", fuzz)
            initial_cell = initial_cell * (fuzz + 1)
        else:
            initial_cell = [round(x, precision_num) for x in initial_cell]
            initial_cell = np.asarray(initial_cell)
        ss.hidden_state = initial_state
        ss.hidden_cell = initial_cell
        recurrent.global_special_regulation(1)
        recurrent.rnn_cell_index_set(rnn_cell_index)
        model_weights = model.get_weights()
        model_config = model.get_config()
        new_initial_model = Model.from_config(model_config)
        new_initial_model.set_weights(model_weights)
        new_result = new_initial_model.predict(x_back_part)
        return original_result, new_result
    elif len(x_test) == 2:
        if rnn_cell_index == 1:
            xq_test = x_test[1]
            x_test = x_test[0]
            original_result = model.predict([x_test.reshape(1, 552), xq_test.reshape(1, 5)])
            layer_index = return_layer_index(model.layers, layer_name)
            new_model = Model(inputs=model.input, outputs=model.layers[layer_index].output)
            input_length = len(x_test)
            x_front_part = x_test[:time_stop_step].reshape(1, time_stop_step)
            x_back_part = x_test[time_stop_step:].reshape(1, input_length - time_stop_step)
            x_front_result = new_model.predict([x_front_part, xq_test])
            initial_state = x_front_result[1][0]
            initial_cell = x_front_result[2][0]
            if op == 1:
                initial_cell = initial_cell * 0
            elif op == 3:
                fuzz = np.random.normal(loc=0.0, scale=standard_deviation, size=None)
                initial_cell = initial_cell * (fuzz + 1)
            else:
                initial_cell = [round(x, precision_num) for x in initial_cell]
                initial_cell = np.asarray(initial_cell)
            ss.hidden_state = initial_state
            ss.hidden_cell = initial_cell
            recurrent.global_special_regulation(1)
            recurrent.rnn_cell_index_set(rnn_cell_index)
            model_weights = model.get_weights()
            model_config = model.get_config()
            new_initial_model = Model.from_config(model_config)
            new_initial_model.set_weights(model_weights)
            new_result = new_initial_model.predict([x_back_part, xq_test])
            return original_result, new_result
        elif rnn_cell_index == 2:
            xq_test = x_test[1]
            x_test = x_test[0]
            original_result = model.predict([x_test.reshape(1, 552), xq_test.reshape(1, 5)])
            layer_index = return_layer_index(model.layers, layer_name)
            new_model = Model(inputs=model.input, outputs=model.layers[layer_index].output)
            input_length = len(xq_test)
            x_front_part = xq_test[:time_stop_step].reshape(1, time_stop_step)
            x_back_part = xq_test[time_stop_step:].reshape(1, input_length - time_stop_step)
            x_front_result = new_model.predict([x_test, x_front_part])
            initial_state = x_front_result[1][0]
            initial_cell = x_front_result[2][0]
            if op == 1:
                initial_cell = initial_cell * 0
            elif op == 3:
                fuzz = np.random.normal(loc=0.0, scale=standard_deviation, size=None)
                initial_cell = initial_cell * (fuzz + 1)
            else:
                initial_cell = [round(x, precision_num) for x in initial_cell]
                initial_cell = np.asarray(initial_cell)
            ss.hidden_state = initial_state
            ss.hidden_cell = initial_cell
            recurrent.global_special_regulation(1)
            recurrent.rnn_cell_index_set(rnn_cell_index)
            model_weights = model.get_weights()
            model_config = model.get_config()
            new_initial_model = Model.from_config(model_config)
            new_initial_model.set_weights(model_weights)
            new_result = new_initial_model.predict([x_test, x_back_part])
            return original_result, new_result


def stateful_state_status_clear(model, x_test, layer_name, batch_size=0):
    """

    :param model:
    :param x_test:
    :param layer_name:
    :param batch_size
    :return:
    """
    units_number = model.get_layer(layer_name).get_config()['units']
    states_list = [0 for i in range(units_number)]
    states_np = np.asarray(states_list)
    states_np = states_np.reshape(1, units_number)
    state_sequence, state_final, cell_final = get_last_state(model, x_test, layer_name, batch_size)
    cell_final = cell_final[-1].reshape(1, units_number)
    layer_index = return_layer_index(model.layers, layer_name)
    if layer_index == 0:
        print("It's not a stateful RNN.")
        return
    else:
        model.layers[layer_index].reset_states(states=[cell_final, states_np])


def stateful_state_reset(model, x_test, layer_name, batch_size=0):
    """

    :param model:
    :param x_test:
    :param layer_name:
    :param batch_size
    :return:
    """
    units_number = model.get_layer(layer_name).get_config()['units']
    state_sequence, state_final, cell_final = get_last_state(model, x_test, layer_name, batch_size)
    state_final = state_final[-1].reshape(1, units_number)
    cell_final = cell_final[-1].reshape(1, units_number)
    print(state_final)
    layer_index = return_layer_index(model.layers, layer_name)
    if layer_index == 0:
        print("It's not a stateful RNN.")
        return
    else:
        # model.layers[layer_index](model.layers[layer_index - 1].output, initial_state=[hidden_state, hidden_cell])
        model.layers[layer_index].reset_states(states=[state_final, cell_final])


def stateful_state_gaussian_fuzzing(model, x_test, layer_name, op, batch_size=0):
    """

    :param model:
    :param x_test:
    :param layer_name:
    :param op:
    :param batch_size:
    :return:
    """
    units_number = model.get_layer(layer_name).get_config()['units']
    state_sequence, state_final, cell_final = get_last_state(model, x_test, layer_name, batch_size)
    if op == 0:
        state_mean = np.mean(state_final[-1])
        state_standard_deviation = np.std(state_final[-1])
        for i in range(len(state_final[-1])):
            state_final[-1][i] = gaussian_fuzzing(state_mean, state_standard_deviation, state_final[-1][i])
    elif op == 1:
        cell_mean = np.mean(cell_final[-1])
        cell_standard_deviation = np.std(cell_final[-1])
        for i in range(len(cell_final[-1])):
            cell_final[-1][i] = gaussian_fuzzing(cell_mean, cell_standard_deviation, cell_final[-1][i])
    state_final = state_final[-1].reshape(1, units_number)
    cell_final = cell_final[-1].reshape(1, units_number)
    layer_index = return_layer_index(model.layers, layer_name)
    if layer_index == 0:
        print("It's not a stateful RNN.")
        return
    else:
        # model.layers[layer_index](model.layers[layer_index - 1].output, initial_state=[hidden_state, hidden_cell])
        model.layers[layer_index].reset_states(states=[cell_final, state_final])


def stateful_state_precision_reduction(model, x_test, layer_name, precision_num, op, batch_size=0):
    """

    :param model:
    :param x_test:
    :param layer_name:
    :param precision_num:
    :param op:
    :param batch_size:
    :return:
    """
    units_number = model.get_layer(layer_name).get_config()['units']
    state_sequence, state_final, cell_final = get_last_state(model, x_test, layer_name, batch_size)
    if op == 0:
        for i in range(len(state_final[-1])):
            state_final[-1][i] = round(state_final[-1][i], precision_num)
    elif op == 1:
        for i in range(len(cell_final[-1])):
            cell_final[-1][i] = round(cell_final[-1][i], precision_num)
    state_final = state_final[-1].reshape(1, units_number)
    cell_final = cell_final[-1].reshape(1, units_number)
    hidden_state = K.variable(value=state_final)
    hidden_cell = K.variable(value=cell_final)
    layer_index = return_layer_index(model.layers, layer_name)
    if layer_index == 0:
        print("It's not a stateful RNN.")
        return
    else:
        model.layers[layer_index](model.layers[layer_index - 1].output, initial_state=[hidden_state, hidden_cell])


def static_gate_process(model, op, standard_deviation, precision_num):
    """

    :param model:
    :param op:
    :param standard_deviation:
    :param precision_num
    :return:
    """
    recurrent.global_set(op)
    recurrent.global_deviation_set(standard_deviation)
    recurrent.global_precision_set(precision_num)

    new_weights = model.get_weights()
    new_config = model.get_config()
    new_model = Model.from_config(new_config)
    new_model.set_weights(new_weights)
    # new_model = Model(inputs=model.input, outputs=model.output)
    # new_model.save("/Users/krogq/mutation_tool/models/test.h5")

    from .utils import data_preprocess, model_QC
    (x_train, y_train), (x_test, y_test) = data_preprocess("imdb")
    # new_model = load_model("/Users/krogq/mutation_tool/models/imdb_lstm.h5")
    y_predict = new_model.predict(x_test.reshape(len(x_test), 80))
    correct_num = 0
    for i in range(len(y_predict)):
        if abs(y_test[i] - y_predict[i][0]) <= 0.5:
            correct_num += 1

    print("acc: ", float(correct_num) / len(y_predict) * 100)


def dynamic_gate_process(model, layer_name, x_test, time_stop_step, op, standard_deviation=0.0):
    """

    :param model:
    :param layer_name:
    :param x_test:
    :param time_stop_step:
    :param op:
    :param standard_deviation:
    :return:
    """
    layer_index = return_layer_index(model.layers, layer_name)
    new_model = Model(inputs=model.input, outputs=model.layers[layer_index].output)
    input_length = len(x_test)

    x_front_part = x_test[:time_stop_step].reshape(1, time_stop_step)
    x_middle = x_test[time_stop_step].reshape(1, 1)
    x_back_part = x_test[time_stop_step + 1:].reshape(1, input_length - time_stop_step - 1)
    # x_front_part = x_test[:time_stop_step].reshape(1, time_stop_step)
    # x_back_part = x_test[time_stop_step:].reshape(1, input_length - time_stop_step)

    x_front_result = new_model.predict(x_front_part)
    initial_state = x_front_result[1][0]
    initial_cell = x_front_result[2][0]
    # save the state in front of the time step
    ss.hidden_state = initial_state
    ss.hidden_cell = initial_cell
    recurrent.global_special_regulation(1)
    recurrent.rnn_cell_index_set(1)
    recurrent.global_set(op)
    recurrent.global_deviation_set(standard_deviation)
    model_weights = new_model.get_weights()
    middle_model_config = new_model.get_config()
    middle_model = Model.from_config(middle_model_config)
    middle_model.set_weights(model_weights)
    x_middle_result = middle_model.predict(x_middle)
    final_initial_state = x_middle_result[1][0]
    final_initial_cell = x_middle_result[2][0]
    # save state after gate process
    ss.hidden_state = final_initial_state
    ss.hidden_cell = final_initial_cell
    recurrent.global_special_regulation(1)
    recurrent.rnn_cell_index_set(1)
    recurrent.global_set(0)
    model_weights = model.get_weights()
    model_config = model.get_config()
    new_initial_model = Model.from_config(model_config)
    new_initial_model.set_weights(model_weights)
    new_result = new_initial_model.predict(x_back_part)
    # del new_model
    # del new_initial_model
    # K.clear_session()
    # gc.collect()
    return new_result


def weights_gaussian_fuzzing(weights, ratio, standard_deviation=0.5, precision_num=2):
    """

    :param weights:
    :param ratio:
    :param standard_deviation:
    :param precision_num:
    :return:
    """
    weights_shape = weights.shape
    flatten_weights = weights.flatten()
    mean = np.mean(flatten_weights)
    std = np.std(flatten_weights)
    weights_len = len(flatten_weights)
    weights_select = np.random.choice(weights_len, int(weights_len * ratio), replace=False)

    for index in weights_select:
        fuzz = np.random.normal(loc=0.0, scale=standard_deviation, size=None)
        # flatten_weights[index] = gaussian_fuzzing(mean, std, flatten_weights[index])
        flatten_weights[index] = flatten_weights[index] * (1 + fuzz)
    flatten_weights = np.clip(flatten_weights, -1.0, 1.0)
    weights = flatten_weights.reshape(weights_shape)
    return weights


def weights_precision_reduction(weights, ratio, precision_num, standard_deviation=0.5):
    """

    :param weights:
    :param ratio:
    :param precision_num:
    :return:
    """
    weights_shape = weights.shape
    flatten_weights = weights.flatten()
    weights_len = len(flatten_weights)
    change_len = int(weights_len * ratio)
    if change_len < 1:
        change_len = 1
    weights_select = np.random.choice(weights_len, change_len, replace=False)
    for index in weights_select:
        flatten_weights[index] = round(flatten_weights[index], precision_num)
    weights = flatten_weights.reshape(weights_shape)
    return weights


def weights_binary_quantization(weights, ratio, standard_deviation=0.5, precision_num=2):
    """

    :param weights:
    :param ratio:
    :param precision_num:
    :return:
    """
    weights_shape = weights.shape
    flatten_weights = weights.flatten()
    weights_len = len(flatten_weights)
    change_len = int(weights_len * ratio)
    if change_len < 1:
        change_len = 1
    weights_select = np.random.choice(weights_len, change_len, replace=False)
    for index in weights_select:
        flatten_weights[index] = np.sign(flatten_weights[index])
    weights = flatten_weights.reshape(weights_shape)
    return weights


def weights_process(model, layer_name, gate_type, ratio, func, standard_deviation=0.5, precision_num=2):
    """

    :param model:
    :param layer_name:
    :param gate_type:
    :param ratio:
    :param func:
    :param standard_deviation:
    :param precision_num:
    :return:
    """
    layer_weights = model.get_layer(layer_name).get_weights()
    input_weights = layer_weights[0]
    state_weights = layer_weights[1]
    bias = layer_weights[2]
    units_number = model.get_layer(layer_name).get_config()['units']
    if gate_type == 0:
        print(standard_deviation)
        input_weights[:, :units_number] = func(input_weights[:, :units_number], ratio, standard_deviation=standard_deviation, precision_num=precision_num)
        state_weights[:, :units_number] = func(state_weights[:, :units_number], ratio, standard_deviation=standard_deviation, precision_num=precision_num)
        bias[:units_number] = func(bias[:units_number], ratio, standard_deviation=standard_deviation, precision_num=precision_num)

    elif gate_type == 1:
        input_weights[:, units_number:  units_number * 2] = func(
            input_weights[:, units_number:  units_number * 2], ratio, precision_num, standard_deviation=standard_deviation, precision_num=precision_num)
        state_weights[:, units_number: units_number * 2] = func(
            state_weights[:, units_number: units_number * 2], ratio, precision_num, standard_deviation=standard_deviation, precision_num=precision_num)
        bias[units_number: units_number * 2] = func(bias[units_number: units_number * 2], ratio, precision_num, standard_deviation=standard_deviation, precision_num=precision_num)

    elif gate_type == 2:
        input_weights[:, units_number * 2: units_number * 3] = func(
            input_weights[:, units_number * 2: units_number * 3], ratio, precision_num, standard_deviation=standard_deviation, precision_num=precision_num)
        state_weights[:, units_number * 2: units_number * 3] = func(
            state_weights[:, units_number * 2: units_number * 3], ratio, precision_num, standard_deviation=standard_deviation, precision_num=precision_num)
        bias[units_number * 2: units_number * 3] = func(bias[units_number * 2: units_number * 3],
                                                                            ratio, precision_num, standard_deviation=standard_deviation, precision_num=precision_num)

    elif gate_type == 3:
        input_weights[:, units_number * 3:] = func(
            input_weights[:, units_number * 3:], ratio, precision_num, standard_deviation=standard_deviation, precision_num=precision_num)
        state_weights[:, units_number * 3:] = func(
            state_weights[:, units_number * 3:], ratio, precision_num, standard_deviation=standard_deviation, precision_num=precision_num)
        bias[units_number * 3:] = func(
            bias[units_number * 3:], ratio, precision_num, standard_deviation=standard_deviation, precision_num=precision_num)

    elif gate_type == 4:
        input_weights = func(input_weights, ratio, precision_num=precision_num, standard_deviation=standard_deviation)
        state_weights = func(state_weights, ratio, precision_num=precision_num, standard_deviation=standard_deviation)
        bias = func(bias, ratio, precision_num=precision_num, standard_deviation=standard_deviation)

    layer_weights[0] = input_weights
    layer_weights[1] = state_weights
    layer_weights[2] = bias
    model.get_layer(layer_name).set_weights(layer_weights)


def gaussian_fuzzing(mean, std, weight):
    """
        gaussian fuzzing for one weight
        :param mean: mean of whole weights of one layer
        :param std: standard deviation of whole weights of one layer
        :param weight: change weight
        :return: weight after fuzzing
        """
    return (1 / np.sqrt(2 * np.pi) * std) * np.exp(-(weight - mean) ** 2 / (2 * std ** 2))


def return_operator_name(int_operator):
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


def return_lstm_gate_name(int_gate):
    """

    :param int_gate:
    :return:
    """
    if int_gate == 0:
        return "input"
    elif int_gate == 1:
        return "forget"
    elif int_gate == 2:
        return "cell"
    elif int_gate == 3:
        return "output"
    else:
        return "all_gate"

