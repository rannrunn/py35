import torch
from torch.autograd import Variable
import numpy as np
import pandas as pd
import random
import math
import os
import matplotlib.pyplot as plt
import copy


# todo: GPU 세팅
# if torch.cuda.is_available():
#     x = x.cuda()

# todo - Question: forget gate 랑 output gate 랑 하는 일이 중복되는 것 아닌가 ㅡ
# todo: 이걸 고려한 것이 GRU 인가 ...?

# todo: layer 를 여러개 쌓는 것은 lstm 에서 어떻게 하지 ...?

# todo: bidirectional lstm 구현 ...?

# todo: many-to-one (1개, 여러개), many-to-many

# todo: rnn & gru 구현


def head(data_x, n):
    if type(data_x) == pd.DataFrame:
        print(data_x.head(n))
    elif type(data_x) == list:
        print(data_x[0:10])


def plot(title, data_x, color="xkcd:azure", is_show=True, is_saveimg=False, save_file_name=''):
    plt.title(title)
    plt.plot(data_x, color=color)
    plt.grid(True)

    if is_saveimg:
        plt.figure(figsize=(8,4))
        plt.savefig('C:\\_data\\lstm_sine_plot\\' + save_file_name + '.png', bbox_inches='tight')

    if is_show:
        plt.show()




def set_dir(path_):
    os.chdir(path_)
    print(os.getcwd())


def get_dir():
    print(os.getcwd())


def normalize(data, min_=-1, max_=1):
    data_normalized = copy.deepcopy(data)
    max_data = max(data_normalized)
    min_data = min(data_normalized)

    for idx in range(len(data)):
        data_normalized[idx] = min_ + ((data_normalized[idx] - min_data) * (max_ - min_)) / (max_data - min_data)

    return data_normalized, max_data, min_data


def de_normalize(data_normalized, max_data, min_data, max_set, min_set):
    data = copy.deepcopy(data_normalized)
    for idx in range(len(data)):
        data[idx] = min_data + ((data[idx]-min_set) * (max_data-min_data))/(max_set-min_set)
    return data


def print_param(param):
    params = []
    for key in vars(param).keys():
        params.append(round(getattr(param, key).data[0], 2))
    print(params)


def print_grad(param):
    for key in vars(param).keys():
        if getattr(param, key).grad is not None:
            print(round(getattr(param, key).grad.data[0], 5))
        else:
            print("None")


def parameters(param):
    for key in vars(param).keys():
        yield getattr(param, key)


def zero_grad(param):
    """Sets gradients of all model parameters to zero."""
    for p in parameters(param):
        if p.grad is not None:
            if p.grad.volatile:
                p.grad.data.zero_()
            else:
                data = p.grad.data
                p.grad = Variable(data.new().resize_as_(data).zero_())

    return param


def mm(w, x):
    if type(x) == int:
        return w * x
    elif len(x) == 1:
        return w * x
    else:
        return torch.mm(w, x)


def numpy_to_var(np_arr, requires_grad=False):
    # type of 'np_arr': numpy.ndarray
    return Variable(torch.from_numpy(np_arr), requires_grad=requires_grad)


def list_to_variable(list_, requires_grad=False):
    data_x_tensor = torch.Tensor(list_)
    return Variable(data_x_tensor, requires_grad=requires_grad)


def rand_arr(*args, min_, max_):
    return np.random.rand(*args) * (max_ - min_) + min_
# def rand_arr(min_, max_, *args):
#     return np.random.rand(*args) * (max_ - min_) + min_


def sigmoid(x):
    return 1. / (1+torch.exp(-x))


def tanh(x):
    return torch.tanh(x)


def get_n_row(data_x):
    if type(data_x) is list:
        return len(data_x)

    if type(data_x) is pd.DataFrame:
        return data_x.shape[0]

    if type(data_x) == torch.autograd.variable.Variable:
        return data_x.shape[0]


def get_list_type_data(data_x, n_row_input, n_row_output):
    # get x
    data_for_rnn_x = []
    n_row = get_n_row(data_x)
    for idx in range(n_row - n_row_input + 1 - n_row_output):
        cnt = 0
        one_row_x = []
        while True:
            one_row_x.append(data_x[idx + cnt])
            if len(one_row_x) == n_row_input:
                one_row_x = list_to_variable(one_row_x, requires_grad=False).type(torch.DoubleTensor)
                data_for_rnn_x.append(one_row_x)
                break
            cnt += 1

    # get y
    data_for_rnn_y = []
    for idx in range(n_row_input, n_row):
        cnt = 0
        one_row_y = []
        while True:
            one_row_y.append(data_x[idx])
            if len(one_row_y) == n_row_output:
                one_row_y = list_to_variable(one_row_y, requires_grad=False).type(torch.DoubleTensor)
                data_for_rnn_y.append(one_row_y)
                break
            cnt += 1

    return data_for_rnn_x, data_for_rnn_y


class Param:
    # todo: 변수 이름들 보기 변하게 바꿈
    def __init__(self):
        # todo: 2) backward 는 autograd 에 맡겨봄

        # todo: 행렬 또는 벡터 크기
        # todo: cell 에만 layer 여러개 쌓는 건가 ...?
        self.weight_x_for_forget_gate = None
        self.weight_output_for_forget_gate = None
        self.bias_for_forget_gate = None

        self.weight_x_for_cell = None
        self.weight_output_for_cell = None
        self.bias_for_cell = None

        self.weight_x_for_input_gate = None
        self.weight_output_for_input_gate = None
        self.bias_for_input_gate = None

        self.weight_x_for_output_gate = None
        self.weight_output_for_output_gate = None
        self.bias_for_output_gate = None

        self.initialize_weights()
        self.change_weight_to_var()

    def initialize_weights(self, min_=-0.1, max_=0.1):
        # todo: size ㅡ 관련된 것들 x, h 등 에 따라 자동 조절 ㅡ 향후 코드 교체
        size = 1
        self.weight_x_for_forget_gate = rand_arr(size, min_=min_, max_=max_)
        self.weight_output_for_forget_gate = rand_arr(size, min_=min_, max_=max_)
        self.bias_for_forget_gate = rand_arr(size, min_=min_, max_=max_)

        self.weight_x_for_cell = rand_arr(size, min_=min_, max_=max_)
        self.weight_output_for_cell = rand_arr(size, min_=min_, max_=max_)
        self.bias_for_cell = rand_arr(size, min_=min_, max_=max_)

        self.weight_x_for_input_gate = rand_arr(size, min_=min_, max_=max_)
        self.weight_output_for_input_gate = rand_arr(size, min_=min_, max_=max_)
        self.bias_for_input_gate = rand_arr(size, min_=min_, max_=max_)

        self.weight_x_for_output_gate = rand_arr(size, min_=min_, max_=max_)
        self.weight_output_for_output_gate = rand_arr(size, min_=min_, max_=max_)
        self.bias_for_output_gate = rand_arr(size, min_=min_, max_=max_)

    def change_weight_to_var(self):
        self.weight_x_for_forget_gate = numpy_to_var(self.weight_x_for_forget_gate, requires_grad=True).type(torch.DoubleTensor)
        self.weight_output_for_forget_gate = numpy_to_var(self.weight_output_for_forget_gate, requires_grad=True).type(torch.DoubleTensor)
        self.bias_for_forget_gate = numpy_to_var(self.bias_for_forget_gate, requires_grad=True).type(torch.DoubleTensor)

        self.weight_x_for_cell = numpy_to_var(self.weight_x_for_cell, requires_grad=True).type(torch.DoubleTensor)
        self.weight_output_for_cell = numpy_to_var(self.weight_output_for_cell, requires_grad=True).type(torch.DoubleTensor)
        self.bias_for_cell = numpy_to_var(self.bias_for_cell, requires_grad=True).type(torch.DoubleTensor)

        self.weight_x_for_input_gate = numpy_to_var(self.weight_x_for_input_gate, requires_grad=True).type(torch.DoubleTensor)
        self.weight_output_for_input_gate = numpy_to_var(self.weight_output_for_input_gate, requires_grad=True).type(torch.DoubleTensor)
        self.bias_for_input_gate = numpy_to_var(self.bias_for_input_gate, requires_grad=True).type(torch.DoubleTensor)

        self.weight_x_for_output_gate = numpy_to_var(self.weight_x_for_output_gate, requires_grad=True).type(torch.DoubleTensor)
        self.weight_output_for_output_gate = numpy_to_var(self.weight_output_for_output_gate, requires_grad=True).type(torch.DoubleTensor)
        self.bias_for_output_gate = numpy_to_var(self.bias_for_output_gate, requires_grad=True).type(torch.DoubleTensor)


def forward_calc(x, output_before, cell_before_forget_gate, param):
    # todo - question: h size ...? 일단 1로 고정하고 시작 ...?

    forget_gate_logit = mm(param.weight_x_for_forget_gate, x) + mm(param.weight_output_for_forget_gate, output_before) + param.bias_for_forget_gate
    forget_gate = sigmoid(forget_gate_logit)

    cell_logit = mm(param.weight_x_for_cell, x) + mm(param.weight_output_for_cell, output_before) + param.bias_for_cell
    cell_before_input_gate = tanh(cell_logit)

    input_gate_logit = mm(param.weight_x_for_input_gate, x) + mm(param.weight_output_for_cell, output_before) + param.bias_for_input_gate
    input_gate = sigmoid(input_gate_logit)

    cell = cell_before_forget_gate * forget_gate + cell_before_input_gate * input_gate

    output_gate_logit = mm(param.weight_x_for_output_gate, x) + mm(param.weight_output_for_output_gate, output_before) + param.bias_for_output_gate
    output_gate = sigmoid(output_gate_logit)

    # todo: regression 에서 output 에 tanh 또는 sigmoid 값 곱하면 ㅡ 잘 안되지 않나 ...?
    output = output_gate * tanh(cell)

    return cell, output


def lstm_forward_propagation(data, param, verbose=False):
    length = get_n_row(data)

    outputs = []

    for idx in range(length):

        if idx == 0:
            # 처음 input 이 들어가면 아래 것들을 0 으로 세팅
            # output = 0
            # cell = 0
            output = data[0]    # todo: bidirectional RNN(or LSTM) 의 경우 ㅡ 마지막 데이터가 들어가게 함
            cell = data[0]

        cell, output = forward_calc(x=data[idx], output_before=output, cell_before_forget_gate=cell, param=param)
        outputs.append(output)

        if verbose is True:
            print("Cell:  ", cell, "\nOutput:", output, "\n")

    return output, outputs


def isnan(x):
    # check nan on torch
    # https://github.com/pytorch/pytorch/issues/4767
    res = x != x
    if res.data[0] is 0:
        return False
    else:
        return True


def lstm_forward_propagation_with_missing_data(data, param, verbose=False):
    length = get_n_row(data)

    outputs = []

    for idx in range(length):

        if idx == 0:
            # 처음 input 이 들어가면 아래 것들을 0 으로 세팅
            # output = 0
            # cell = 0
            output = data[0]    # todo: bidirectional RNN(or LSTM) 의 경우 ㅡ 마지막 데이터가 들어가게 함
            cell = data[0]

        if isnan(data[idx]):
            cell, output = forward_calc(x=output, output_before=output, cell_before_forget_gate=cell, param=param)
        else:
            cell, output = forward_calc(x=data[idx], output_before=output, cell_before_forget_gate=cell, param=param)
        outputs.append(output)

        if verbose is True:
            print("Cell:  ", cell, "\nOutput:", output, "\n")

    return output, outputs


def loss_fn(y, pred):
    # todo: 일단 output size 1, 1 로 생각하고 ㅡ 나중에 바꿈

    # todo: loss function 은 mse 로 일단 하고 ㅡ 나중에 바꿈

    loss = (y - pred).pow(2)
    return loss
