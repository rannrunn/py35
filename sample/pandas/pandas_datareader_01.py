import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import datetime

start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2016, 12, 31)

df_KOSPI = web.DataReader("KRX:KOSPI", "google", start, end)
df_SE = web.DataReader("KRX:005930", "google", start, end)
df_SKH = web.DataReader("KRX:000660", "google", start, end)

plt.figure(figsize  = (8,6))
plt.scatter(df_SE["Close"], df_KOSPI["Close"])
plt.show()

plt.figure(figsize  = (8,6))
plt.scatter(df_SKH["Close"], df_KOSPI["Close"])
plt.show()


