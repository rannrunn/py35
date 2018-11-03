# coding: utf-8
import sys
sys.setrecursionlimit(10000)
import pandas as pd
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_columns', 30)
import os
import numpy as np


class Tree(object):
    def __init__(self, df_dl=None, df_sec=None, df_sw_frtu=None, dict_tree_info=None, dict_sw_info=None):
        self.df_dl = df_dl
        self.df_sec = df_sec
        self.df_sw_frtu = df_sw_frtu
        self.sw_flag = dict_sw_info['sw_flag']
        self.sw_id = dict_sw_info['sw_id']
        self.sw_loc = dict_sw_info['sw_loc']
        self.dl_id = dict_sw_info['dl_id']
        self.sw_id_f = dict_sw_info['sw_id_f']
        self.list_detail_sw_id = []
        self.list_detail_sw_link_info = []
        self.children = []

        self.depth = dict_tree_info['depth']


    def __repr__(self):
        sw_link_info = ''
        if self.sw_flag == '1' or self.sw_flag == '2':
            list_sw = []
            str_link_list = ''
            # tuple_sw[0] : 스위치 아이디
            # tuple_sw[1] : 스위치 연결 정보 dictionary
            for tuple_sw in self.list_detail_sw_link_info:
                list_sw.append(tuple_sw[0])
                str_link_list += ', ' + str(tuple_sw[0]) + ':F' + str(tuple_sw[1]['f']) + ':B' + str(tuple_sw[1]['b'])
            str_link_list = '(' + str_link_list[2:] + ')'
            if self.sw_flag == '1':
                sw_id_or_loc = str(self.sw_id)
            elif self.sw_flag == '2':
                sw_id_or_loc = self.sw_loc
            sw_link_info = sw_id_or_loc.ljust(30, ' ') + ' ' + str(list_sw).ljust(50, ' ') + ' ' + str_link_list
        elif self.sw_flag == '3':
            sw_link_info = 'Blank'.ljust(30, ' ') + ' ' + str([]).ljust(50, ' ') + ' ' + '(Blank:F[' + str(self.sw_id_f) + ']:B[])'
        result = '[' + str(self.dl_id).ljust(3, ' ') + '] ' + str(self.depth).ljust(4) + sw_link_info
        return str(result)


    def add_detail_sw_id(self, sr_sw_id):
        if self.sw_flag == '1':
            self.list_detail_sw_id.append(self.sw_id)
        elif self.sw_flag == '2':
            if sr_sw_id is not None:
                for item_sw_id in sr_sw_id:
                    self.list_detail_sw_id.append(item_sw_id)
        elif self.sw_flag == '3':
            pass


    def set_detail_sw_link_info(self):
        for val_sw_id in self.list_detail_sw_id:
            list_link_forward = []
            list_link_backward = []
            for sw_id_f in self.df_sec[df_sec['sw_id_b'] == val_sw_id]['sw_id_f']:
                list_link_forward.append(sw_id_f)
            for sw_id_b in self.df_sec[df_sec['sw_id_f'] == val_sw_id]['sw_id_b']:
                list_link_backward.append(sw_id_b)
            tuple_sw = (val_sw_id, dict({'f':list_link_forward, 'b':list_link_backward}))
            self.list_detail_sw_link_info.append(tuple_sw)


    def add_child(self, children):
        if children is not None:
            for child in children:
                assert isinstance(child, Tree)
                self.children.append(child)


class DL(object):
    def __init__(self, df_dl=None, df_sec=None, df_sw_frtu=None, dl_id=None):

        self.df_dl = df_dl[df_dl['dl_id'] == dl_id]
        self.dl_id = dl_id
        self.dl_name = self.df_dl.loc[df_dl['dl_id'] == dl_id, 'dl_name'].values[0]
        self.cb_id = self.df_dl.loc[df_dl['dl_id'] == dl_id, 'cb_id'].values[0]
        self.dl_list_sw = []
        self.dl_tree = self.set_dl_tree_list(df_dl, df_sec, df_sw_frtu, self.set_dict_tree_info(0), self.set_dict_sw_info('1', self.cb_id, '', self.dl_id, np.nan))
        # self.dl_sw_count


    def __repr__(self):
        return self.dl_id + '_' + self.dl_name


    def get_dl_list_sw(self):
        return self.dl_list_sw


    def set_dict_tree_info(self, depth):
        return dict({'depth':depth})


    def det_tree_info(self, parent_depth):
        depth = parent_depth + 1
        return dict({'depth':depth})


    def set_dict_sw_info(self, sw_flag, sw_id, sw_loc, dl_id, sw_id_f):
        return dict({'sw_flag':sw_flag, 'sw_id':sw_id, 'sw_loc':sw_loc, 'dl_id':dl_id, 'sw_id_f':sw_id_f})


    def set_sw_info(self, sw_id, df_sw_frtu, dl_id, sw_id_f):
        result = dict()

        sw_flag = ''
        sw_loc = ''
        sw_dl_id = dl_id
        sw_id_f = sw_id_f

        df_local_sw_frtu = df_sw_frtu[(df_sw_frtu['sw_id'] == sw_id) & (df_sw_frtu['sw_id'].notnull())]
        sr_sw_id = pd.DataFrame()
        sw_loc = ''
        if not df_local_sw_frtu.empty:
            sw_dl_id = df_local_sw_frtu['dl_id'].values[0]
            sw_loc = df_local_sw_frtu['sw_loc'].values[0]
            sw_loc = sw_loc[:sw_loc.find('(')] if sw_loc.find('(') > -1 else sw_loc
            sr_sw_id = df_sw_frtu[df_sw_frtu['sw_loc'].apply(lambda x: x[:x.find('(')] if x.find('(') > -1 else x) == sw_loc]['sw_id']

        # 다회로 개폐기의 경우 -> 2
        if len(sr_sw_id) > 1:
            result = self.set_dict_sw_info('2', sw_id, sw_loc, sw_dl_id, sw_id_f)
        # 싱글 개폐기 중 스위치 id 가 없는 경우 -> 3
        elif np.isnan(sw_id):
            result = self.set_dict_sw_info('3', sw_id, sw_loc, sw_dl_id, sw_id_f)
        # 싱글 개폐기의 경우 -> 1
        else:
            result = self.set_dict_sw_info('1', sw_id, sw_loc, sw_dl_id, sw_id_f)
        return result


    def set_dl_tree_list(self, df_dl, df_sec, df_sw_frtu, dict_tree_info, dict_sw_info):
        children = []
        # sw_flag 1:싱글 개폐기, 2:다회로 개폐기, 3:블랭크
        sw_flag = dict_sw_info['sw_flag']
        sw_id = dict_sw_info['sw_id']
        sw_loc = dict_sw_info['sw_loc']
        sw_dl_id = dict_sw_info['dl_id']
        sw_id_f = dict_sw_info['sw_id_f']
        tree = Tree(df_dl, df_sec, df_sw_frtu, dict_tree_info, dict_sw_info)

        dict_next_tree_info = self.set_dict_tree_info(dict_tree_info['depth'] + 1)

        if sw_flag == '1':

            tree.add_detail_sw_id(pd.Series())
            tree.set_detail_sw_link_info()

            # dl_list_sw 에 sw_id 가 있을 경우 dl_list_sw 에 sw_id 추가하고 리턴
            if sw_id in self.dl_list_sw:
                self.dl_list_sw.append(sw_id)
                return tree
            self.dl_list_sw.append(sw_id)

            # 다른 DL의 개폐기인 경우
            if not np.isnan(sw_dl_id) and sw_dl_id != self.dl_id:
                return tree

            for sw_id_b in df_sec[df_sec['sw_id_f'] == sw_id]['sw_id_b']:
                # dict_sw_info 의 sw_id_f 는 for 문에서 나온 sw_id_b 이다.
                dict_next_sw_info = self.set_sw_info(sw_id_b, df_sw_frtu, sw_dl_id, sw_id)
                children.append(self.set_dl_tree_list(df_dl, df_sec, df_sw_frtu, dict_next_tree_info, dict_next_sw_info))

        elif sw_flag == '2':

            sr_sw_id = df_sw_frtu[df_sw_frtu['sw_loc'].apply(lambda x: x[:x.find('(')] if x.find('(') > -1 else x) == sw_loc]['sw_id']
            tree.add_detail_sw_id(sr_sw_id)
            tree.set_detail_sw_link_info()

            # dl_list_sw 에 sw_loc 이 있을 경우 리턴 dl_list_sw 에 sw_loc 추가하고 리턴
            if sw_loc in self.dl_list_sw:
                self.dl_list_sw.append(sw_loc)
                return tree
            self.dl_list_sw.append(sw_loc)

            # 다른 DL의 개폐기인 경우
            if not np.isnan(sw_dl_id) and sw_dl_id != self.dl_id:
                return tree

            for sw_id_detail in sr_sw_id:
                for sw_id_b in df_sec[df_sec['sw_id_f'] == sw_id_detail]['sw_id_b']:
                    # dict_sw_info 의 sw_id_f 는 for 문에서 나온 sw_id_detail 이다.
                    dict_next_sw_info = self.set_sw_info(sw_id_b, df_sw_frtu, sw_dl_id, sw_id_detail)
                    children.append(self.set_dl_tree_list(df_dl, df_sec, df_sw_frtu, dict_next_tree_info, dict_next_sw_info))

        elif sw_flag == '3':

            # dl_list_sw 에 스위치 아이디가 없는 항목이 있든 없든 dl_list_sw 에 추가하고 리턴
            self.dl_list_sw.append(sw_id)
            return tree

        tree.add_child(children)

        return tree


    def search(self, Tree):
        print(Tree)
        if Tree.children is not None:
            for child in Tree.children:
                self.search(child)


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
        print('트리 시작')
        dl.search(dl.dl_tree)
        print('트리 종료')

        # df_temp = pd.DataFrame([[dl_id, dl_name, cb_id, len(list_dl)]], columns=['DL_ID', 'DL_NAME', 'CB_ID', 'COUNT'])
        # df_dl_line_count = df_dl_line_count.append(df_temp, ignore_index=True)

        print('idx: ' + str(idx) + ', dl_id: ' + str(dl_id) + ', dl_name: ' + dl_name)

    if not os.path.isdir(dir):
        os.makedirs(dir)
    df_dl_line_count.to_csv(os.path.join(dir_distribution_line, 'dl_sw_count.csv'), index=False)

