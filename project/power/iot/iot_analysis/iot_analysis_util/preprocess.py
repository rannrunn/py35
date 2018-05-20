import numpy as np

# 센서 오작동 데이터 제거 : 제작사 코드가 1인 센서에서만 오작동 센서 데이터가 있었지만 전체 센서에 적용
# 제작사 코드 1
# 오류1 : 온도(300), 습도(300), 피치(500), 롤(500), 조도(60000), 자외선(200), 대기압(100)
# 오류2 : 온도(300), 피치(500), 롤(500)
# 오류3 : 습도(300), 조도(60000), 자외선(200), 대기압(100)
def remove_sensor_malfunction_data(df):
    if 'TEMP' in df.columns:
        df = df[(df['TEMP'] == 300) == False]
    if 'HUMI' in df.columns:
        df = df[(df['HUMI'] == 300) == False]
    return df

# 데이터 범위 : 일단 전 제작사 공통 범위 사용하고, 범위를 벗어나는 데이터는 np.nan 으로 변환
# 온도 : -40 ~125
# 습도 : 0 ~ 100
# 피치 : -180 ~ 180
# 롤 : -90 ~ 90
# 조도 : 0 ~ 60000
# 자외선 : 0 ~ 20.48
# 대기압 : 300 ~ 1100
# 베터리 : 0 ~ 100 (1차:전압을퍼센트로환산, 2차:전압)
# 주기 : 처리 안함
# 전류 : 처리 안함
# 충격 : 처리 안함
# 자기장 X : -5000 ~ 5000
# 자기장 Y : -5000 ~ 5000
# 자기장 Z : -5000 ~ 5000
# 진동 X : -16000 ~ 16000
# 진동 Y : -16000 ~ 16000
# 진동 Z : -16000 ~ 16000
# 초음파 : 0~3,000
# 유중온도 : -20 ~ 120
# 자외선-C : 0 ~ 5 (스펙상 0~5 사이이나 실제 5를 초과하는 데이터도 측정됨)
def remove_out_of_range_data(df):
    if 'TEMP' in df.columns:
        df['TEMP'] = df['TEMP'].apply(lambda x: x if -40 <= x and x <= 125 else np.nan)
    if 'HUMI' in df.columns:
        df['HUMI'] = df['HUMI'].apply(lambda x: x if 0 <= x and x <= 100 else np.nan)
    if 'PITCH' in df.columns:
        df['PITCH'] = df['PITCH'].apply(lambda x: x if -180 <= x and x <= 180 else np.nan)
    if 'ROLL' in df.columns:
        df['ROLL'] = df['ROLL'].apply(lambda x: x if -90 <= x and x <= 90 else np.nan)
    if 'AMBIENT' in df.columns:
        df['AMBIENT'] = df['AMBIENT'].apply(lambda x: x if 0 <= x and x <= 60000 else np.nan)
    if 'UV' in df.columns:
        df['UV'] = df['UV'].apply(lambda x: x if 0 <= x and x <= 20.48 else np.nan)
    if 'BATTERY' in df.columns:
        df['BATTERY'] = df['BATTERY'].apply(lambda x: x if 0 <= x and x <= 100 else np.nan)
    if 'GEOMAG_X' in df.columns:
        df['GEOMAG_X'] = df['GEOMAG_X'].apply(lambda x: x if -5000 <= x and x <= 5000 else np.nan)
    if 'GEOMAG_Y' in df.columns:
        df['GEOMAG_Y'] = df['GEOMAG_Y'].apply(lambda x: x if -5000 <= x and x <= 5000 else np.nan)
    if 'GEOMAG_Z' in df.columns:
        df['GEOMAG_Z'] = df['GEOMAG_Z'].apply(lambda x: x if -5000 <= x and x <= 5000 else np.nan)
    if 'VAR_X' in df.columns:
        df['VAR_X'] = df['VAR_X'].apply(lambda x: x if -16000 <= x and x <= 16000 else np.nan)
    if 'VAR_Y' in df.columns:
        df['VAR_Y'] = df['VAR_Y'].apply(lambda x: x if -16000 <= x and x <= 16000 else np.nan)
    if 'VAR_Z' in df.columns:
        df['VAR_Z'] = df['VAR_Z'].apply(lambda x: x if -16000 <= x and x <= 16000 else np.nan)
    if 'USN' in df.columns:
        df['USN'] = df['USN'].apply(lambda x: x if 0 <= x and x <= 3000 else np.nan)
    if 'NTC' in df.columns:
        df['NTC'] = df['NTC'].apply(lambda x: x if -20 <= x and x <= 120 else np.nan)
    if 'UVC' in df.columns:
        df['UVC'] = df['UVC'].apply(lambda x: x if 0 <= x and x <= 5 else np.nan)

    return df