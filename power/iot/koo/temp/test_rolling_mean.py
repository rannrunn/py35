import pandas as pd
import numpy as np


df = pd.DataFrame([1.0, 2.0, 4.0, np.nan, 2.0, np.nan], columns=['A'])
df = df.rolling(2).mean()
print(df)


