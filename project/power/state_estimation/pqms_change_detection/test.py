import numpy as np
import pandas as pd
from pandas.core.nanops import nanmean as pd_nanmean
from statsmodels.tsa.seasonal import DecomposeResult
from statsmodels.tsa.filters._utils import _maybe_get_pandas_wrapper_freq
import statsmodels.api as sm

observed = np.array([i for i in range(10, 20, 1)])

detrended = np.array([i for i in range(10)])

period = 2


period_averages = np.array([pd_nanmean(detrended[i::period]) for i in range(period)])
# 0-center the period avgs
# 평균을 배열로 만들고 다시 평균을 계산
period_averages -= np.mean(period_averages)
#
seasonal = np.tile(period_averages, len(observed) // period + 1)[:len(observed)]