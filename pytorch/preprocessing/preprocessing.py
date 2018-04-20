import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import copy
import torch
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler


def clip_gradient(grad_data, clip_val=1):
    """
    grad: gradient
    val: value

    example:
    w.grad.data = clip_gradient(w.grad.data)
    """
    _grad_val = grad_data[0]
    _grad_norm = abs(_grad_val)

    if abs(_grad_val) > 1:
        _grad_val = (clip_val/_grad_norm) * _grad_val

    grad_data[0] = _grad_val
    return grad_data


def get_normal_data(mu_, sigma_, n_=1000, dim_=2, is_return_tensor_var=False):
    """
    tensor_var: tensor Variable
    """

    """
    data = get_normal_data(mu_=1, sigma_=2, n_=1000)
    plt.plot(data[:, 0], data[:, 1], 'go')
    plt.show()
    """
    _data = np.random.normal(mu_, sigma_, (n_, dim_))

    if is_return_tensor_var:
        # 1) change 'np array' to 'torch tensor'
        # 2) set 'variable type' as 'torch.DoubleTensor'
        _data = torch.from_numpy(_data).type(torch.DoubleTensor)

        _data = Variable(_data, requires_grad=False)
        print("\nrequires_grad:", _data.requires_grad)
    return _data


def head(data_, n_=3):
    if isinstance(data_, np.ndarray):
        _res = data_[0:n_, :]
    elif isinstance(data_, torch.Variable):
        _res = data_[0:n_, :]
    else:
        raise NotImplementedError("not np.ndarray...!")
    return _res


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def scale_minus1_to_1(data_):
    _scale = MinMaxScaler(feature_range=(-1, 1))  # MinMax Scaler
    data_ = _scale.fit_transform(data_)  # input: ndarray type data
    return data_, _scale


def scale_0_to_1(data_):
    _scale = MinMaxScaler(feature_range=(0, 1))  # MinMax Scaler
    data_ = _scale.fit_transform(data_)  # input: ndarray type data
    return data_, _scale


def df_to_torch(df_, torch_type_=torch.FloatTensor, requires_grad_=True):
    df_tensor_ = torch.from_numpy(np.asarray(df_)).type(torch_type_)
    df_tensor_ = Variable(df_tensor_, requires_grad=requires_grad_)

    return df_tensor_


class SplitData:
    def __init__(self, data_, train_percent_=None, split_date_=None):
        self._data = data_
        self._train_percent = train_percent_
        self._split_date = split_date_

    def x_y(self, y_col=None):
        """
        example:
            X_train, y_train = split_X_y(train_data, "Species")
            X_test, y_test = split_X_y(test_data, "Species")
        """
        if y_col is None:
            raise NotImplementedError("col num or name is required...!")
        elif isinstance(y_col, int):
            y_data = self._data.ix[:, y_col]
            _x_data = self._data.drop(self._data.columns[y_col], axis=1)
        elif isinstance(y_col, str):
            y_data = self._data.ix[:, y_col]
            _x_data = self._data.drop(y_col, axis=1)
        return _x_data, y_data

    def split(self):
        train_size = int(self._data.shape[0] * self._train_percent)
        train_data_, test_data_ = self._data.ix[0:train_size, :], self._data.ix[train_size:n, :]
        print("\n", "train length:", train_data_.shape[0], "\n", "test length:", test_data_.shape[0])
        train_data_.drop((train_data_.shape[0] - 1), axis=0, inplace=True)
        return train_data_, test_data_

    def randomly(self):
        n = self._data.shape[0]

        train_idx = np.random.choice(np.arange(0, n), int(n*self._train_percent), replace=False).tolist()

        train_data_, test_data_ = self._data.ix[train_idx, :], self._data.drop(train_idx, axis=0)
        print("\n", "train length:", train_data_.shape[0], "\n", "test length:", test_data_.shape[0])

        train_data_.reset_index(drop=True, inplace=True)
        test_data_.reset_index(drop=True, inplace=True)
        return train_data_, test_data_

    def by_date(self):
        if not isinstance(self._data, pd.DataFrame):
            raise NotImplementedError("pd.DataFrame is required...!")

        split_date = pd.Timestamp(self._split_date)

        train_data_ = self._data.ix[:split_date, :]
        train_data_.drop(split_date, axis=0, inplace=True)

        test_data_ = self._data.ix[split_date:, :]
        return train_data_, test_data_


def get_iris():
    _iris = load_iris()

    _x = pd.DataFrame(_iris.data)
    _x.columns = _iris.feature_names

    _y = pd.DataFrame(_iris.target)
    _y.columns = ["Species"]

    _data = pd.concat([_x, _y], axis=1)
    return _data
