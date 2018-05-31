import sys
sys.setrecursionlimit(10000)
import pandas as pd
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_columns', 30)
import os
import math
import numpy as np


class Tree(object):
    def __init__(self, sw_flag=None, sw_id=None, sw_loc=None, dl_id=None):
        self.sw_flag = sw_flag
        self.sw_id = sw_id
        self.sw_loc = sw_loc
        self.dl_id = dl_id
        self.children = []

    def __repr__(self):
        result = ''
        if self.sw_flag == '1':
            result = self.sw_id
        elif self.sw_flag == '2':
            result = self.sw_loc
        elif self.sw_flag == '3':
            result = 'Blank'
        return

    def add_child(self, children):
        if children is not None:
            for child in children:
                assert isinstance(child, Tree)
                self.children.append(child)

    def search(self, Tree):
        if Tree.sw_flag == '1':
            print(Tree.sw_id)
        elif Tree.sw_flag == '2':
            print(Tree.sw_loc)
        elif Tree.sw_flag == '3':
            print('노스위치')
        if Tree.children is not None:
            for child in Tree.children:
                Tree.search(child)


class DL(object):
    def __init__(self, df_dl=None, df_sec=None, df_sw_frtu=None, dl_id=None):


        self.dl_id = dl_id
        self.df_dl = df_dl[df_dl['dl_id'] == dl_id]
        self.dl_name = self.df_dl['dl_name'].values[0]
        self.cb_id = self.df_dl['cb_id'].values[0]
        self.dl_list_sw = []
        self.dl_tree = self.set_dl_tree_list(df_dl, df_sec, df_sw_frtu, self.set_dict_sw_info('1', self.cb_id, '', self.dl_id))
        self.dl_tree.search(self.dl_tree)
        # self.dl_sw_count


    def __repr__(self):
        return self.dl_id + '_' + self.dl_name


    def get_dl_list_sw(self):
        return self.dl_list_sw


    def set_dict_sw_info(self, sw_flag, sw_id, sw_loc, dl_id):
        return dict({'sw_flag':sw_flag, 'sw_id':sw_id, 'sw_loc':sw_loc, 'dl_id':dl_id})


    def set_sw_info(self, sw_id, df_sw_frtu):
        result = dict()

        sw_flag = ''
        sw_id = sw_id
        sw_loc = ''
        sw_dl_id = ''

        df_local_sw_frtu = df_sw_frtu[(df_sw_frtu['sw_id'] == sw_id) & (df_sw_frtu['sw_id'].notnull())]
        sr_sw_id = pd.DataFrame()
        sw_loc = ''
        if len(df_local_sw_frtu) > 0:
            sw_dl_id = df_local_sw_frtu['dl_id'].values[0]
            sw_loc = df_local_sw_frtu['sw_loc'].values[0]
            sw_loc = sw_loc[:sw_loc.find('(')] if sw_loc.find('(') > -1 else sw_loc
            sr_sw_id = df_sw_frtu[df_sw_frtu['sw_loc'].apply(lambda x: x[:x.find('(')] if x.find('(') > -1 else x) == sw_loc]['sw_id']

        # 다회로 개폐기의 경우 -> 2
        if len(sr_sw_id) > 1:
            result = self.set_dict_sw_info('2', sw_id, sw_loc, sw_dl_id)
        # 싱글 개폐기 중 스위치 id 가 없는 경우 -> 3
        elif sw_id == '':
            result = self.set_dict_sw_info('3', sw_id, sw_loc, sw_dl_id)
        # 싱글 개폐기의 경우 -> 1
        else:
            result = self.set_dict_sw_info('1', sw_id, sw_loc, sw_dl_id)
        return result


    # return list: 연결정보 있고 목적 스위치가 있음, None: 연결정보 없음(진짜 없는 경우 or DL이 다른 경우), Blank: 연결정보 있으나 목적 스위치가 없음
    def set_dl_tree_list(self, df_dl, df_sec, df_sw_frtu, dict_sw_info):
        children = []
        # sw_flag 1:싱글 개폐기, 2:다회로 개폐기
        sw_flag = dict_sw_info['sw_flag']
        sw_id = dict_sw_info['sw_id']
        sw_loc = dict_sw_info['sw_loc']
        sw_dl_id = dict_sw_info['dl_id']
        tree = Tree(sw_flag, sw_id, sw_loc, sw_dl_id)

        if sw_flag == '1':
            if sw_id in self.dl_list_sw:
                self.dl_list_sw.append(sw_id)
                return tree
            self.dl_list_sw.append(sw_id)
        elif sw_flag == '2':
            if sw_loc in self.dl_list_sw:
                self.dl_list_sw.append(sw_loc)
                return tree
            self.dl_list_sw.append(sw_loc)

        # 다른 DL의 개폐기 이거나 스위치 아이디가 없는 경우
        if sw_dl_id != '' and sw_dl_id != self.dl_id:
            return tree
        # 스위치 아이디가 없는 경우
        if sw_id == '':
            return tree

        if sw_flag == '1':
            for sw_id_b in df_sec[df_sec['sw_id_f'] == sw_id]['sw_id_b']:
                dict_sw_info = self.set_sw_info(sw_id_b, df_sw_frtu)
                children.append(self.set_dl_tree_list(df_dl, df_sec, df_sw_frtu, dict_sw_info))
        elif sw_flag == '2':
            sr_sw_id = df_sw_frtu[df_sw_frtu['sw_loc'].apply(lambda x: x[:x.find('(')] if x.find('(') > -1 else x) == sw_loc]['sw_id']
            for sw_id_detail in sr_sw_id:
                for sw_id_b in df_sec[df_sec['sw_id_f'] == sw_id_detail]['sw_id_b']:
                    dict_sw_info = self.set_sw_info(sw_id_b, df_sw_frtu)
                    children.append(self.set_dl_tree_list(df_dl, df_sec, df_sw_frtu, dict_sw_info))

        tree.add_child(children)

        return tree


class SW(object):
    def __init__(self):
        pass


class CB(SW):
    def __init__(self, cb_id=None):
        # print('CB')
        self.cb_id = cb_id
        self.children = self.get_node(self.cb_id, False)

    def __repr__(self):
        return self.cb_id


class SW_SINGLE(SW):
    def __init__(self, sw_id_f=None):
        # print('SW_SINGLE')
        self.sw_id_f = sw_id_f
        self.children = self.get_node(self.sw_id_f, False)

    def __repr__(self):
        return self.sw_id_f


class SW_MULTI(SW):
    def __init__(self, sw_id_f=None):
        # print('SW_MULTI')
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
    # DL 9 는 SW_FRTU 테이블에 개폐기가 여러 개 있으나 연결이 끊기니 구현된 단선도 프로그램을 확인해 봐야함
    # 18 : 다회로 개폐기 4개

    df_dl_line_count = pd.DataFrame(columns=['DL_ID', 'DL_NAME', 'CB_ID', 'COUNT'])

    for idx in range(len(df_dl)):

        dl_id = df_dl.loc[idx, 'dl_id']
        dl_name = df_dl.loc[idx, 'dl_name']

        if dl_id != 23:
            continue

        cb_id = None
        if len(df_dl.loc[df_dl['dl_id'] == dl_id, 'cb_id']) > 0:
            cb_id = df_dl.loc[df_dl['dl_id'] == dl_id, 'cb_id'].values[0]
        else:
            print('dl_id가 없습니다.')

        print('dl_id:', dl_id)
        if dl_id is not None and cb_id is not None:
            dl = DL(df_dl, df_sec, df_sw_frtu, dl_id)

        print('ALL SWITCH:', dl.get_dl_list_sw())
        print(len(list(set(dl.get_dl_list_sw()))))
        print(list(set(dl.get_dl_list_sw())))

        # df_temp = pd.DataFrame([[dl_id, dl_name, cb_id, len(list_dl)]], columns=['DL_ID', 'DL_NAME', 'CB_ID', 'COUNT'])
        # df_dl_line_count = df_dl_line_count.append(df_temp, ignore_index=True)

        print('idx: ' + str(idx) + ', dl_id: ' + str(dl_id) + ', dl_name: ' + dl_name)

    if not os.path.isdir(dir):
        os.makedirs(dir)
    df_dl_line_count.to_csv(os.path.join(dir_distribution_line, 'dl_sw_count.csv'), index=False)

