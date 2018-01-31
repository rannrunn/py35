import pandas as pd
import numpy as np


df = pd.DataFrame([[1.0, 2.0, 4.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], columns=['A', 'B', 'C'])
print(df)
print(np.max(df.max()))
print(np.mean(df.mean()))
print(np.max(df.max()) - np.min(df.min()))
print(df - df.mean())
df_norm = (df - np.mean(df.mean())) / (np.max(df.max()) - np.min(df.min()))

print(df_norm)