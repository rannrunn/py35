import pandas as pd

df = pd.DataFrame()

# with 구문을 사용하여 프린트
with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
    print(df)


# to_string() 사용하여 프린트
print(df.to_string())


# 옵션을 사용하여 프린트
def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')


# 옵션을 사용하여 프린트 2
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
