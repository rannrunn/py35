import numpy as np
np.set_printoptions(threshold=np.nan)
import pandas as pd
import matplotlib.pyplot as plt


df = pd.DataFrame(np.random.randn(10, 2))

print(df)

ran = np.random.rand(df.shape[0])
print(ran)

df[ran > 0.5] = 1.5

print(df)

df = df.replace(1.5, np.nan)


print(df)

print(df.fillna(0))



