import sys
sys.setrecursionlimit(10000)
import pandas as pd
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_columns', 30)
import os


def function(dl_id, sw_id_f, multi_flag, list_sw):
    # 같은 sw가 SEC 테이블에서 들어온 경우('.0'이 있음)와 SW_FRTU 테이블에서 들어온 경우('.0'이 없음)를 구분하여 조건을 따짐
    # (의도한 것은 아니지만 구현이 필요한 알고리즘이 자동으로 해결되었음)
    if str(sw_id_f) in list_sw and str(sw_id_f) != 'nan':
        # print(str(sw_id_f) + '가 반복됩니다.')
        return list_sw
    # 다중화 개폐기에 속한 개폐기 중 인풋 개폐기를 제외한 나머지 개폐기들은 리스트에 추가시키지 않음
    if multi_flag == False:
        list_sw.append(str(sw_id_f))
    if str(sw_id_f) == 'nan':
        return list_sw
    df_local_sec = df_sec.loc[df_sec['sw_id_f'] == sw_id_f]
    df_local_sw_frtu = df_sw_frtu[(df_sw_frtu['dl_id'] == dl_id) & (df_sw_frtu['sw_id'] == sw_id_f)]
    if len(df_local_sec) != 0:
        sw_id_b = df_local_sec['sw_id_b'].values[0]
        # print('sw_id_f:', sw_id_f, 'sw_id_b:', sw_id_b)
        function(dl_id, sw_id_b, False, list_sw)
    # 다중화 개폐기의 인풋이 바로 아웃풋인 경우가 있는 지 확인해 봐야 함
    elif len(df_local_sw_frtu) > 0 and multi_flag == False:
        sw_loc = df_local_sw_frtu['sw_loc'].values[0]
        sw_loc = sw_loc[:sw_loc.find('(')] if sw_loc.find('(') > -1 else sw_loc
        sr_sw_id = df_sw_frtu[df_sw_frtu['sw_loc'].apply(lambda x: x[:x.find('(')] if x.find('(') > -1 else x) == sw_loc]['sw_id']
        # print('개폐기 인풋:', sw_id_f)
        # print('개폐기 로케이션:', sw_loc)
        # print('개폐기 세부 개폐기:')
        # for val in sr_sw_id:
        #     print(val)
        # print('개폐기 끝')
        for val in sr_sw_id:
            function(dl_id, val, True, list_sw)
        # print('있음')
    return list_sw


if __name__ == '__main__':

    x1 = pd.ExcelFile('C:\\_data\\das_data.xls')

    # print(x1.sheet_names)

    df_dl = x1.parse('dl')
    df_sec = x1.parse('sec')
    df_sw_frtu = x1.parse('sw_frtu')


    # print(df_dl.head())
    # print(df_sec.head())
    # print(df_sw_frtu.head())


    # DL 6 의 마지막 SW 인 26555 는 SW_FRTU 테이블에 DL 11에 속한다고 되어 있으니 전력연구원에 질문해야 할 듯 함
    # DL 8, 10 은 루프가 있어 오류남
    # DL 9 는 SW_FRTU 테이블에 개폐기가 여러개 있으나 연결이 끊기니 구현된 단선도 프로그램을 확인해 봐야함
    # 18 : 다중화 개폐기 4개


    df_dl_line_count = pd.DataFrame(columns=['DL_ID', 'DL_NAME', 'CB_ID', 'COUNT'])
    for idx in range(len(df_dl)):

        dl_id = df_dl.loc[idx, 'dl_id']
        dl_name = df_dl.loc[idx, 'dl_name']

        cb_id = None
        if len(df_dl.loc[df_dl['dl_id'] == dl_id, 'cb_id']) > 0:
            cb_id = df_dl.loc[df_dl['dl_id'] == dl_id, 'cb_id'].values[0]
        else:
            print('dl_id가 없습니다.')
            continue

        list_sw = []
        if dl_id is not None and cb_id is not None:
            function(dl_id, cb_id, False, list_sw)

        for idx_2, val in  enumerate(list_sw):
            list_sw[idx_2] = list_sw[idx_2].replace('.0', '')
        list_sw = list(set(list_sw))

        df_temp = pd.DataFrame([[dl_id, dl_name, cb_id, len(list_sw)]], columns=['DL_ID', 'DL_NAME', 'CB_ID', 'COUNT'])
        df_dl_line_count = df_dl_line_count.append(df_temp, ignore_index=True)

        print('idx:', idx)

    dir = 'C:\\_data'
    if not os.path.isdir(dir):
        os.makedirs(dir)
    df_dl_line_count.to_csv(os.path.join(dir, 'dl_line_count.csv'), index=False)

