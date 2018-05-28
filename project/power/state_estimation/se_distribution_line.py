import sys
sys.setrecursionlimit(300)
import pandas as pd
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_columns', 30)


def function(dl_id, sw_id_f, multi_flag):
    df_local_sec = df_sec.loc[df_sec['sw_id_f'] == sw_id_f]
    df_local_sw_frtu = df_sw_frtu[(df_sw_frtu['dl_id'] == dl_id) & (df_sw_frtu['sw_id'] == sw_id_f)]
    if len(df_local_sec) != 0:
        sw_id_b = df_local_sec['sw_id_b'].values[0]
        print('sw_id_f:', sw_id_f, 'sw_id_b:', sw_id_b)
        function(dl_id, sw_id_b, False)
    # 다중화 개폐기의 인풋이 바로 아웃풋인 경우는 없다고 생각
    elif len(df_local_sw_frtu) > 0 and multi_flag == False:
        sw_loc = df_local_sw_frtu['sw_loc'].values[0]
        sw_loc = sw_loc[:sw_loc.find('(')] if sw_loc.find('(') > -1 else sw_loc
        sr_sw_id = df_sw_frtu[df_sw_frtu['sw_loc'].apply(lambda x: x[:x.find('(')] if x.find('(') > -1 else x) == sw_loc]['sw_id']
        print('다중화개폐기 인풋:', sw_id_f)
        print('다중화개폐기 로케이션:', sw_loc)
        print('다중화개폐기 세부 개폐기:')
        for val in sr_sw_id:
            print(val)
        print('다중화개폐기 끝')
        for val in sr_sw_id:
            function(dl_id, val, True)
        # print('있음')
    return


if __name__ == '__main__':
    x1 = pd.ExcelFile('C:\\_data\\das_data.xls')

    print(x1.sheet_names)

    df_dl = x1.parse('dl')
    df_sec = x1.parse('sec')
    df_sw_frtu = x1.parse('sw_frtu')

    print(df_dl.head())
    print(df_sec.head())
    print(df_sw_frtu.head())

    # DL 8, 10 은 루프가 있어 오류남
    dl_id = 10

    cb_id = None
    if len(df_dl.loc[df_dl['dl_id'] == dl_id, 'cb_id']) > 0:
        cb_id = df_dl.loc[df_dl['dl_id'] == dl_id, 'cb_id'].values[0]
    else:
        print('dl_id가 없습니다.')

    print('dl_id:', dl_id)
    print('cb_id:', cb_id)

    if dl_id is not None and cb_id is not None:
        function(dl_id, cb_id, False)