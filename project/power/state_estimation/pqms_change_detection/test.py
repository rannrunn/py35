df_H_data = []

df_H_data['datetime'] = df_H_data.index.values
df_H_data['weekday'] = df_H_data['datetime'].dt.weekday
df_H_data['day_of_year'] = df_H_data['datetime'].dt.dayofyear
df_H_data['day_of_years'] = df_H_data['datetime'].apply(lambda x: df_H_data.loc[x, 'day_of_year'] + sum([datetime.date(item, 12, 31).timetuple().tm_yday for item in list(set(df_H_data['datetime'].dt.year.values)) if item < df_H_data.loc[x, 'datetime'].year]))


df_H_data['date_group_week'] = df_H_data['datetime'].apply(lambda x: df_H_data.loc[x, 'day_of_years'] - (df_H_data.loc[x, 'weekday'] + 1))
df_H_data = df_H_data[(df_H_data['date_group_week'] != df_H_data['date_group_week'].min()) & (df_H_data['date_group_week'] != df_H_data['date_group_week'].max())]

df_H_data['data_week_odd'] = df_H_data.loc[df_H_data['date_group_week'] % 2 == 0, 'load']
df_H_data['data_week_even'] = df_H_data.loc[df_H_data['date_group_week'] % 2 == 1, 'load']


df_H_mean_des_week = pd.DataFrame(df_H_data.groupby('date_group_week')['load'].describe())
df_H_mean_des_week['date_group_week'] = df_H_mean_des_week.index
df_H_mean_des_week.rename(columns={'count': 'week_count', 'mean': 'week_mean', 'std': 'week_std', 'min': 'week_min', '25%': 'week_25%', '50%': 'week_50%', '75%': 'week_75%', 'max': 'week_max', 'yyyymmdd': 'yyyymmdd'}, inplace=True)
df_H_data = pd.merge(df_H_data, df_H_mean_des_week, on='date_group_week', how='left')

df_H_data = df_H_data