from tensorflow.keras.models import Sequential
import numpy as np
import random
from tensorflow import keras
from utils import get_type_layers, find_sameshape_layer, summary_model


def cnn_operator(model, operator, ratio, standard_deviation=0.5):
    """

    :param model:
    :param operator: GF = 0
                     WS = 1
                     NEB = 2
                     NAI = 3
                     NS = 4
                     LR = 5
                     LA = 6
                     LD = 7
    :param ratio:
    :param standard_deviation:
    :return:
    """
    dense_layer_list, convolution_layer_list, dense_con_layer_list, flatten_layer_list = get_type_layers(model)
    weight_count, neuron_count, weights_dict, neuron_dict = summary_model(model)
    process_weights_num = int(weight_count * ratio) if int(weight_count * ratio) > 0 else 1
    process_neuron_num = int(neuron_count * ratio) if int(neuron_count * ratio) > 0 else 1
    # print(len(weight_count))
    # print(process_weights_num)
    if operator == 0:
        # GF
        process_num_list = random_select(weight_count, process_weights_num, dense_con_layer_list, weights_dict)
        for layer_index in range(len(dense_con_layer_list)):
            if process_num_list[layer_index] == 0:
                continue
            layer_name = dense_con_layer_list[layer_index]
            l_weights = model.get_layer(layer_name).get_weights()
            new_l_weights = weights_gaussian_fuzzing(l_weights, process_num_list[layer_index], standard_deviation)
            model.get_layer(layer_name).set_weights(new_l_weights)

    elif operator == 1:
        # WS
        process_num_list = random_select(neuron_count, process_neuron_num, dense_con_layer_list, neuron_dict)
        for layer_index in range(len(dense_con_layer_list)):
            if process_num_list[layer_index] == 0:
                continue
            layer_name = dense_con_layer_list[layer_index]
            l_weights = model.get_layer(layer_name).get_weights()
            new_l_weights = weights_shuffle(l_weights, process_num_list[layer_index])
            model.get_layer(layer_name).set_weights(new_l_weights)

    elif operator == 2:
        # NEB
        process_num_list = random_select(neuron_count, process_neuron_num, dense_con_layer_list, neuron_dict)
        for layer_index in range(len(dense_con_layer_list)):
            if process_num_list[layer_index] == 0:
                continue
            layer_name = dense_con_layer_list[layer_index]
            l_weights = model.get_layer(layer_name).get_weights()
            new_l_weights = neural_delete_random(l_weights, process_num_list[layer_index])
            model.get_layer(layer_name).set_weights(new_l_weights)
    elif operator == 3:
        # NAI
        process_num_list = random_select(neuron_count, process_neuron_num, dense_con_layer_list, neuron_dict)
        for layer_index in range(len(dense_con_layer_list)):
            if process_num_list[layer_index] == 0:
                continue
            layer_name = dense_con_layer_list[layer_index]
            l_weights = model.get_layer(layer_name).get_weights()
            new_l_weights = neuron_activation_inverse(l_weights, process_num_list[layer_index])
            model.get_layer(layer_name).set_weights(new_l_weights)
    elif operator == 4:
        # NS
        process_num_list = random_select(neuron_count, process_neuron_num, dense_con_layer_list, neuron_dict)
        for layer_index in range(len(dense_con_layer_list)):
            if process_num_list[layer_index] == 0:
                continue
            layer_name = dense_con_layer_list[layer_index]
            l_weights = model.get_layer(layer_name).get_weights()
            new_l_weights = neuron_switch(l_weights, process_num_list[layer_index])
            model.get_layer(layer_name).set_weights(new_l_weights)
    elif operator == 5:
        # LR
        candidate_layer_list = find_sameshape_layer(model)
        if len(candidate_layer_list) == 0:
            print("No such layer.")
            return False
        else:
            layer_select = random.sample(1, len(candidate_layer_list))
            new_model = layer_remove(model, candidate_layer_list[layer_select])
            return new_model
    elif operator == 6:
        # LA
        add_position = random.randint(1, len(model.layers))
        new_model = activation_layer_addition(model, add_position)
        return new_model
    elif operator == 7:
        # LD
        candidate_layer_list = find_sameshape_layer(model)
        if len(candidate_layer_list) == 0:
            print("No such layer.")
        else:
            layer_select = random.sample(1, len(candidate_layer_list))
            new_model = duplicate_layer_addition(model, candidate_layer_list[layer_select])
            return new_model


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


def weights_gaussian_fuzzing(weights, process_num, standard_deviation=0.5):
    """

    :param weights:
    :param process_num:
    :param standard_deviation:
    :return:
    """
    weights = weights.copy()
    layer_weights = weights[0]
    weights_shape = layer_weights.shape
    flatten_weights = layer_weights.flatten()
    weights_len = len(flatten_weights)
    weights_select = np.random.choice(weights_len, process_num, replace=False)
    for index in weights_select:
        fuzz = np.random.normal(loc=0.0, scale=standard_deviation, size=None)
        flatten_weights[index] = flatten_weights[index] * (1 + fuzz)
    flatten_weights = np.clip(flatten_weights, -1.0, 1.0)
    weights[0] = flatten_weights.reshape(weights_shape)
    return weights


def weights_shuffle(weights, process_num):
    """

    :param weights:
    :param process_num:
    :return:
    """
    weights = weights.copy()
    layer_weights = weights[0].T
    neural_num = len(layer_weights)
    neural_select = random.sample(range(0, neural_num - 1), process_num)
    weights_shape = layer_weights[0].shape
    for neural_index in neural_select:
        flatten_weights = layer_weights[neural_index].flatten()
        np.random.shuffle(flatten_weights)
        flatten_weights = flatten_weights.reshape(weights_shape)
        layer_weights[neural_index] = flatten_weights
    weights[0] = layer_weights.T
    return weights


def neuron_activation_inverse(weights, process_num):
    """
    neuron activation inverse
    weights = -1 * weights
    bias = -1 * bias
    :param weights: one layer weights
    :param process_num:
    :return: mutant model
    """
    weights = weights.copy()
    layer_weights = weights[0].T
    bias = weights[1]
    weights_len = len(layer_weights)
    neural_select = random.sample(range(0, weights_len - 1), process_num)
    weights_shape = layer_weights[0].shape
    for neural_index in neural_select:
        flatten_weights = layer_weights[neural_index].flatten()
        flatten_weights = np.array([-x for x in flatten_weights])
        flatten_weights = flatten_weights.reshape(weights_shape)
        layer_weights[neural_index] = flatten_weights
        bias[neural_index] = -1 * bias[neural_index]
    weights[0] = layer_weights.T
    weights[1] = bias
    return weights


def neural_delete_random(weights, process_num):
    """
    random delete one neural
    weights -> 0
    bias -> 0
    :param weights: layer weights
    :param process_num: mutation ratio
    :return: mutant model
    """
    weights = weights.copy()
    layer_weights = weights[0].T
    bias = weights[1]
    neural_num = len(layer_weights)
    neural_select = random.sample(range(0, neural_num - 1), process_num)
    weights_shape = layer_weights[0].shape
    for neural_index in neural_select:
        layer_weights[neural_index] = np.full(weights_shape, 0)
        bias[neural_index] = 0
    weights[0] = layer_weights.T
    weights[1] = bias
    return weights


def neuron_switch(weights, process_num):
    """

    :param weights:
    :param process_num:
    :return:
    """
    weights = weights.copy()
    layer_weights = weights[0].T
    bias = weights[1]
    neural_num = len(layer_weights)
    select_num = process_num
    for i in range(select_num):
        neuron_pair = random.sample(range(0, neural_num - 1), 2)
        flag = layer_weights[neuron_pair[0]]
        layer_weights[neuron_pair[0]] = layer_weights[neuron_pair[1]]
        layer_weights[neuron_pair[1]] = flag
        flag = bias[neuron_pair[0]]
        bias[neuron_pair[0]] = bias[neuron_pair[1]]
        bias[neuron_pair[1]] = flag
    weights[0] = layer_weights.T
    weights[1] = bias
    return weights


def activation_layer_addition(model, add_position):
    """
    add activation layer
    activation function: tanh, softmax, elu, selu, softplus, linear
                        softsign, relu, sigmoid, hard_sigmoid
    :param model: original model
    :param add_position: selected position
    :return: mutant model
    """
    activation_list = ["tanh", "softmax", "elu", "selu", "softplus", "softsign", "relu", "sigmoid", "hard_sigmoid",
                       "linear"]
    activation_selected = random.sample(activation_list, 1)[0]
    layer_add = keras.layers.Activation(activation_selected)
    layer_len = len(model.layers)
    new_model = Sequential()
    for layer_index in range(layer_len):
        new_model.add(model.layers[layer_index])
        if layer_index == add_position:
            new_model.add(layer_add)
    return new_model


def duplicate_layer_addition(model, layer_name):
    """
    copy one layer
    selected layer must has the same input and output shape
    :param model: original model
    :param layer_name: copy layer selected
    :return: mutant model
    """
    new_model = Sequential()
    for layer in model.layers:
        if layer.name == layer_name:
            layer_insert = layer
            new_model.add(layer)
            new_model.add(layer_insert)
            continue
        new_model.add(layer)
    return new_model


def layer_remove(model, layer_name):
    """
    remove one layer
    selected layer must has the same input and output shape
    :param model: original model
    :param layer_name: layer selected
    :return: mutant model
    """
    new_model = Sequential()
    for layer in model.layers:
        if layer.name == layer_name:
            continue
        new_model.add(layer)
    return new_model


def random_bias_change(model, layer_name, ratio, scale):
    """
    random change one layer bias
    :param model: original bias
    :param layer_name: layer selected
    :param ratio: mutation ratio
    :param scale: change scale
    :return: mutant model
    """
    weights = model.get_layer(layer_name).get_weights()
    bias_len = len(weights[1])
    bias_select = np.random.choice(bias_len, int(bias_len * ratio), replace=False)
    for bias_index in bias_select:
        op = random.randint(0, 2)
        weights[1][bias_index] = bias_change(weights[1][bias_index], op, scale)
    model.get_layer(layer_name).set_weights(weights)
    return model


def bias_change(bias, op, scale):
    """
    change one bias
    :param bias: selected bias
    :param op: operation
                1 : inverse
                2 : zero
                3 : scale
    :param scale: scale ratio
    :return: bias changed
    """
    if op == 0:
        bias = -1 * bias
    elif op == 1:
        bias = 0
    elif op == 2:
        bias = bias * scale
    return bias


def cnn_operator_name(int_operator):
    """

    :param int_operator:
    :return:
    """
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
