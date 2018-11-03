# coding: utf-8

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt


# the main library has a small set of functionality
from stldecompose import decompose, forecast
from stldecompose.forecast_funcs import naive
from stldecompose.forecast_funcs import drift
from stldecompose.forecast_funcs import mean
from stldecompose.forecast_funcs import seasonal_naive

period = 365

# We'll use some of the data that comes pre-packaged with `statsmodels` to demonstrate the library functionality. The data set below comprises incomplete, daily measurements of CO2 levels in Hawaii. 

dataset = sm.datasets.co2.load_pandas()
obs = dataset.data

obs.head()
obs

obs.plot()
plt.show()
plt.close()


# Because it's based on some existing `statsmodels` functionality, `STLDecompose` requires two things of the input dataframe:
# 1. continuous observations (no missing data points)
# 2. a `pandas` `DateTimeIndex`
# 
# Since these are both very situation-dependent, we leave it to the user to define how they want to acheive these goals - `pandas` provides a number of ways [to work with missing data](https://pandas.pydata.org/pandas-docs/stable/missing_data.html). In particular, the functions shown below make these steps relatively straightforward. Below, we add use linear interpolation, and resample to daily observations. The resulting frame meets both of our criteria. 

# interpolate linear -> 미취득값을 선형보간 하는 것
obs = obs.resample('D').mean().interpolate('linear')
obs.head(10)


obs.index


obs.head(1000).plot()
plt.show()
plt.close()

# # Decompose
# 
# One of the primary pieces of functionality is the STL decomposition. The associated method requires the observation frame, and the primary (largest) period of seasonality. This `period` is in terms of index positions, and so care is needed for the user to correctly specify the periodicity in terms of their observations.
# 
# For example, with daily observations and large annual cycles, `period=365`. For hourly observations with large daily cycles, `period=24`. Some inspection, and trial and error may be helpful.

stl = decompose(obs, period=period)



# The resulting object is an extended version of the `statsmodels.tsa.seasonal.DecomposeResult`. Like the `statsmodels` object, the arrays of values are available on the object (the observations; and the trend, seasonal, and residual components). An extra attribute (the avergage seasonal cycle) has been added for the purpose of forecasting. 
# 
# We inherit the built-in `.plot()` method on the object.

stl.plot();
plt.title('stl')
plt.show()
plt.close()


# # Forecast
# 
# While the STL decomposition is interesting on it's own, `STLDecompose` also provides some relatively naive capabilities for using the decomposition to forecast based on our observations. 
# 
# We'll use the same data set, but pretend that we only had the first two third of observations. Then we can compare our forecast to the real observation data. 


len(obs)


short_obs = obs.head(10000)



# apply the decomp to the truncated observation
short_stl = decompose(short_obs, period=period)

short_stl


# The `forecast()` method requires the following arguments:
# - the fit `DecomposeResult`
# - the number of steps forward for which we'd like the forecast
# - the specific forecasting function we'll apply to the decomposition
# 
# There are a handful of predefined functions that can be imported from the `stldecompose.forecast_funcs` module. These implementations are based on [Hyndman's online textbook](https://www.otexts.org/fpp/2/3). The user can also define their own forecast function, following the patterns demonstrated in the predefined functions. 
# 
# The return type of the `forecast()` method is a `pandas.Dataframe` with a column name that represents the forecast function and an appropriate `DatetimeIndex`.

fcast = forecast(short_stl, steps=8000, fc_func=drift)

fcast.head()


# If desired, we can then plot the corresponding components of the observation and forecast to check and verify the results.

plt.plot(obs, '--', label='truth')
plt.plot(short_obs, '--', label='obs')
plt.plot(short_stl.trend, ':', label='stl.trend')
plt.plot(fcast, '-', label=fcast.columns[0])

plt.xlim('1970','2004'); plt.ylim(330,380);
plt.legend();
plt.show()
plt.close()


# To include the estimated seasonal component in the forecast, use the boolean `seasonal` keyword.

fcast = forecast(short_stl, steps=8000, fc_func=drift, seasonal=True)

plt.plot(obs, '--', label='truth')
plt.plot(short_obs, '--', label='obs')
plt.plot(short_stl.trend, ':', label='stl.trend')
plt.plot(fcast, '-', label=fcast.columns[0])

plt.xlim('1970','2004'); plt.ylim(330,380);
plt.legend();


fcast.head()

plt.show()
plt.close()




