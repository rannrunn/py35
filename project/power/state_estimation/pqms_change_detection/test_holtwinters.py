import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

df = pd.read_csv('C:\\international-airline-passengers.csv',
                 parse_dates=['Month'],
                 index_col='Month'
                 )
# df.index.freq = 'MS'
train, test = df.iloc[:90, 0], df.iloc[89:, 0]
model = ExponentialSmoothing(train, seasonal='mul', seasonal_periods=12).fit()
pred = model.predict(start=test.index[0], end=test.index[-1])

plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test')
plt.plot(pred.index, pred, label='Holt-Winters')
plt.plot(train.index, pred, label='Holt-Winters')
plt.legend(loc='best')
plt.show()