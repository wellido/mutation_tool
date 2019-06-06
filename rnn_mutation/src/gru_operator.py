from .lstm_operator import *


def gru_operator(model, layer_name, operation, x_test, rnn_cell_index=1, ratio=0.01, gate_type=0, time_stop_step=0,
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
    :param rnn_cell_index
    :param ratio:
    :param gate_type: 0 - update
                      1 - reset
                      2 - cell candidate
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
            return gru_not_stateful_state_status_process(model, layer_name, rnn_cell_index, x_test, time_stop_step, operation)
    elif operation == 2:
        if stateful:
            stateful_state_reset(model, x_test, layer_name, batch_size)
        else:
            return gru_not_stateful_state_reset(model, layer_name, x_test, time_stop_step, time_start_step)
    elif operation == 3:
        if stateful:
            stateful_state_gaussian_fuzzing(model, x_test, layer_name, 0, batch_size)
        else:
            return gru_not_stateful_state_status_process(model, layer_name, rnn_cell_index, x_test, time_stop_step, operation)
    elif operation == 4:
        if stateful:
            stateful_state_precision_reduction(model, x_test, layer_name, precision_num, 0, batch_size)
        else:
            return gru_not_stateful_state_status_process(model, layer_name, rnn_cell_index, x_test, time_stop_step, operation)
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
    elif operation == 8:
        if gate_type == 0:
            static_gate_process(2, standard_deviation, precision_num)
        elif gate_type == 1:
            static_gate_process(4, standard_deviation, precision_num)
        elif gate_type == 2:
            static_gate_process(6, standard_deviation, precision_num)
    elif operation == 9:
        if gate_type == 0:
            static_gate_process(11, standard_deviation, precision_num)
        elif gate_type == 1:
            static_gate_process(13, standard_deviation, precision_num)
        elif gate_type == 2:
            static_gate_process(15, standard_deviation, precision_num)
    elif operation == 10:
        gru_weights_process(model, layer_name, gate_type, ratio, weights_gaussian_fuzzing, standard_deviation=standard_deviation)
    elif operation == 11:
        weights_process(model, layer_name, gate_type, ratio, weights_binary_quantization)
    elif operation == 12:
        weights_process(model, layer_name, gate_type, ratio, weights_precision_reduction, precision_num=precision_num)


def gru_weights_process(model, layer_name, gate_type, ratio, func, standard_deviation=0.5, precision_num=2):
    """

    :param model:
    :param layer_name:
    :param gate_type:
    :param ratio:
    :param func:
    :param precision_num:
    :return:
    """
    print("standard_deviation: ", standard_deviation)
    print("ratio: ", ratio)
    print("gate_type: ", gate_type)
    layer_weights = model.get_layer(layer_name).get_weights()
    input_weights = layer_weights[0]
    state_weights = layer_weights[1]
    bias = layer_weights[2]
    units_number = model.get_layer(layer_name).get_config()['units']
    if gate_type == 0:
        input_weights[:, : units_number] = func(input_weights[:, : units_number], ratio, precision_num, standard_deviation=standard_deviation)
        state_weights[:, : units_number] = func(state_weights[:, : units_number], ratio, precision_num, standard_deviation=standard_deviation)
        bias[:units_number] = func(bias[:units_number], ratio, precision_num, standard_deviation=standard_deviation)
    elif gate_type == 1:
        input_weights[:, units_number: units_number * 2] = func(input_weights[:, units_number: units_number * 2],
                                                                ratio, precision_num, standard_deviation=standard_deviation)
        state_weights[:, units_number: units_number * 2] = func(state_weights[:, units_number: units_number * 2],
                                                                ratio, precision_num, standard_deviation=standard_deviation)
        bias[units_number: units_number * 2] = func(bias[units_number: units_number * 2], ratio, precision_num, standard_deviation=standard_deviation)
    elif gate_type == 2:
        input_weights[:, units_number * 2:] = func(input_weights[:, units_number * 2:], ratio, precision_num, standard_deviation=standard_deviation)
        state_weights[:, units_number * 2:] = func(state_weights[:, units_number * 2:], ratio, precision_num, standard_deviation=standard_deviation)
        bias[units_number * 2:] = func(bias[units_number * 2:], ratio, precision_num, standard_deviation=standard_deviation)

    elif gate_type == 3:
        input_weights = func(input_weights, ratio, precision_num=precision_num, standard_deviation=standard_deviation)
        state_weights = func(state_weights, ratio, precision_num=precision_num, standard_deviation=standard_deviation)
        bias = func(bias, ratio, precision_num=precision_num, standard_deviation=standard_deviation)
    layer_weights[0] = input_weights
    layer_weights[1] = state_weights
    layer_weights[2] = bias
    model.get_layer(layer_name).set_weights(layer_weights)


def gru_not_stateful_state_reset(model, layer_name, rnn_cell_index, x_test, time_stop_step, time_start_step):
    """

    :param model:
    :param layer_name:
    :param rnn_cell_index
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
        initial_state = x_time_stop_result[1][0]
        x_time_start = x_test[:time_start_step].reshape(1, time_start_step)
        x_time_start_result = new_model.predict(x_time_start)
        initial_cell = x_time_start_result[0][0]
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
            initial_state = x_time_stop_result[1][0]
            x_time_start = x_test[:time_start_step].reshape(1, time_start_step)
            x_time_start_result = new_model.predict([x_time_start, xq_test])
            initial_cell = x_time_start_result[0][0]
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
            initial_state = x_time_stop_result[1][0]
            x_time_start = xq_test[:time_start_step].reshape(1, time_start_step)
            x_time_start_result = new_model.predict([x_test, x_time_start])
            initial_cell = x_time_start_result[0][0]
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


def gru_not_stateful_state_status_process(model, layer_name, rnn_cell_index, x_test, time_stop_step, op,
                                          standard_deviation=1.0, precision_num=2):
    """

    :param model:
    :param layer_name:
    :param rnn_cell_index
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
        initial_state = x_front_result[0][0]
        initial_cell = x_front_result[1][0]
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
            initial_state = x_front_result[0][0]
            initial_cell = x_front_result[1][0]
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
            initial_state = x_front_result[0][0]
            initial_cell = x_front_result[1][0]
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


def return_gru_gate_name(int_gate):
    """

    :param int_gate:
    :return:
    """
    if int_gate == 0:
        return "update"
    elif int_gate == 1:
        return "reset"
    elif int_gate == 2:
        return "candidate"
    else:
        return "all_gate"
