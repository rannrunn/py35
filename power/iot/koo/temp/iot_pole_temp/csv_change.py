import pandas as pd

df = pd.read_csv('./x_train.csv', header=None)
df_y = pd.read_csv('./y_train.csv', header=None)


for item in df.columns.values:
    sr = df[item].notnull()
    df = df[sr]
    df_y = df_y[sr]

print(len(df))
print(len(df_y))

print(df)
print(df_y)

df.to_csv('./x_train_modify.csv', index=False)
df_y.to_csv('./y_train_modify.csv', index=False)