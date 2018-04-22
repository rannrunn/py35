import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm


data = sm.datasets.get_rdataset("co2")
df = data.data

def yearfraction2datetime(yearfraction, startyear=0):
    import datetime, dateutil
    year = int(yearfraction) + startyear
    month = int(round(12 * (yearfraction - year)))
    delta = dateutil.relativedelta.relativedelta(months=month)
    date = datetime.datetime(year, 1, 1) + delta
    return date

df["datetime"] = df.time.map(yearfraction2datetime)
df.tail()


df.plot(x="datetime", y="co2")
plt.show()


df["y"] = df["co2"].diff()
df2 = df.iloc[1:, :]
df2.plot(x="datetime", y="y")
plt.show()


ax1 = plt.subplot(211); sm.graphics.tsa.plot_acf(df2["y"], lags=120, ax=ax1)
ax2 = plt.subplot(212); sm.graphics.tsa.plot_pacf(df2["y"], lags=120, ax=ax2)
plt.tight_layout()
plt.show()


df.loc[:, "month"] = df.datetime.dt.month.values
df.tail()

df[df.month == 1].plot(x="datetime", y="co2", marker='o')
plt.show()


df["y2"] = df["co2"].diff(12)
df3 = df.iloc[12:, :]
df3[df3.month == 1].head(24)

df3[df3.month == 1].plot(x="datetime", y="y2", marker='o')
plt.show()


ax1 = plt.subplot(211); sm.graphics.tsa.plot_acf(df3[df3.month == 1]["y2"], ax=ax1)
ax2 = plt.subplot(212); sm.graphics.tsa.plot_pacf(df3[df3.month == 1]["y2"], ax=ax2)
plt.tight_layout()
plt.show()