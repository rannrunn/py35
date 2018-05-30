import sys
sys.setrecursionlimit(10000)
import pandas as pd
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_columns', 30)
import os


class DL(object):
    def __init__(self, dl_id=None):
        self.dl_id = dl_id
        self.df_dl = df_dl[df_dl['dl_id'] == dl_id]
        self.dl_name = self.set_dl_name()
        self.cb_id = None
        self.set_cb_id()
        print('시작')
        print(self.cb_id)
        self.dl_sw_objects = CB(self.cb_id)
        # self.dl_sw_count

    def __repr__(self):
        return self.dl_id

    def set_dl_name(self):
        print(self.df_dl['dl_name'].values[0])
        self.dl_name = self.df_dl['dl_name'].values[0]

    def set_cb_id(self):
        print(self.df_dl['cb_id'].values[0])
        self.cb_id = self.df_dl['cb_id'].values[0]

class SW(object):
    def get_node(self, sw_id_f, multi_flag):
        result_list_switch = []
        # 같은 sw가 SEC 테이블에서 들어온 경우('.0'이 있음)와 SW_FRTU 테이블에서 들어온 경우('.0'이 없음)를 구분하여 조건을 따짐
        # (의도한 것은 아니지만 구현이 필요한 알고리즘이 자동으로 해결되었음)
        if str(sw_id_f) in list_sw and str(sw_id_f) != 'nan':
            print(str(sw_id_f) + '가 반복됩니다.')
            return list_sw
        # 다회로 개폐기에 속한 개폐기 중 인풋 개폐기를 제외한 나머지 개폐기들은 리스트에 추가시키지 않음
        if multi_flag == False:
            list_sw.append(str(sw_id_f))
        if str(sw_id_f) == 'nan':
            return list_sw
        df_local_sec = df_sec.loc[df_sec['sw_id_f'] == sw_id_f]
        df_local_sw_frtu = df_sw_frtu[(df_sw_frtu['dl_id'] == dl_id) & (df_sw_frtu['sw_id'] == sw_id_f)]
        sr_sw_id = pd.DataFrame()
        if len(df_local_sw_frtu) > 0:
            sw_loc = df_local_sw_frtu['sw_loc'].values[0]
            sw_loc = sw_loc[:sw_loc.find('(')] if sw_loc.find('(') > -1 else sw_loc
            sr_sw_id = df_sw_frtu[df_sw_frtu['sw_loc'].apply(lambda x: x[:x.find('(')] if x.find('(') > -1 else x) == sw_loc]['sw_id']

        # 다회로 개폐기의 인풋이 바로 아웃풋인 경우가 있는 지 확인해 봐야 함 : 없는 듯
        # SW_FRTU 테이블에 DL_ID가 같고, 같은 그룹에 속한 SW 가 1개를 초과할 경우에만 다회로 개폐기 SW 탐색
        # 다회로 개폐기 탐색을 통해 함수가 실행된 경우에 다시 다회로 개폐기 탐색을 할 경우 LOOP 가 발생하므로 이 경우는 패스
        if len(sr_sw_id) > 1 and multi_flag == False:
            print('개폐기 인풋: ' + str(sw_id_f))
            print('개폐기 로케이션: ' + str(sw_loc))
            print('개폐기 세부 개폐기:')
            for sw_id_b in sr_sw_id:
                print(str(sw_id_b))
            print('개폐기 끝')
            for sw_id_b in sr_sw_id:
                result_list_switch.append(SW_MULTI(sw_id_b))
        elif len(df_local_sec) != 0 and ((len(df_sw_frtu[(df_sw_frtu['sw_id'] == sw_id_f)]) == 0) or (len(df_local_sw_frtu) != 0)):
            sr_sw_id_b = df_local_sec['sw_id_b']
            for sw_id_b in sr_sw_id_b:
                print('sw_id_f: ' + str(sw_id_f) + ', sw_id_b: ' + str(sw_id_b))
                result_list_switch.append(SW_SINGLE(sw_id_b))
        return result_list_switch


class CB(SW):
    def __init__(self, cb_id=None):
        print('CB')
        self.cb_id = cb_id
        self.children = self.get_node(self.cb_id, False)

    def __repr__(self):
        return self.cb_id


class SW_SINGLE(SW):
    def __init__(self, sw_id_f=None):
        print('SW_SINGLE')
        self.sw_id_f = sw_id_f
        self.children = self.get_node(self.sw_id_f, False)

    def __repr__(self):
        return self.sw_id_f


class SW_MULTI(SW):
    def __init__(self, sw_id_f=None):
        print('SW_MULTI')
        self.sw_id_f = sw_id_f
        self.children = self.get_node(self.sw_id_f, True)

    def __repr__(self):
        return self.sw_id_f



if __name__ == '__main__':

    dir = 'C:\\_data'
    dir_distribution_line = dir + '\\distribution_line'
    dir_distribution_line_topology = dir_distribution_line + '\\topology'

    x1 = pd.ExcelFile(os.path.join(dir, 'das_data_20180529.xls'))

    if not os.path.isdir(dir_distribution_line_topology):
        os.makedirs(dir_distribution_line_topology)

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
    # 18 : 다회로 개폐기 4개

    df_dl_line_count = pd.DataFrame(columns=['DL_ID', 'DL_NAME', 'CB_ID', 'COUNT'])

    for idx in range(len(df_dl)):

        dl_id = df_dl.loc[idx, 'dl_id']
        dl_name = df_dl.loc[idx, 'dl_name']

        if dl_id != 149:
            continue

        cb_id = None
        if len(df_dl.loc[df_dl['dl_id'] == dl_id, 'cb_id']) > 0:
            cb_id = df_dl.loc[df_dl['dl_id'] == dl_id, 'cb_id'].values[0]
        else:
            print('dl_id가 없습니다.')

        list_sw = []
        if dl_id is not None and cb_id is not None:
            DL(dl_id)

        print(list_sw)

        for idx_2, val in  enumerate(list_sw):
            list_sw[idx_2] = list_sw[idx_2].replace('.0', '')
        list_sw = list(set(list_sw))

        df_temp = pd.DataFrame([[dl_id, dl_name, cb_id, len(list_sw)]], columns=['DL_ID', 'DL_NAME', 'CB_ID', 'COUNT'])
        df_dl_line_count = df_dl_line_count.append(df_temp, ignore_index=True)

        print('idx: ' + str(idx) + ', dl_id: ' + str(dl_id) + ', dl_name: ' + str(dl_name))

    if not os.path.isdir(dir):
        os.makedirs(dir)
    df_dl_line_count.to_csv(os.path.join(dir_distribution_line, 'dl_sw_count.csv'), index=False)

