# 변압기 본체, 전주, 변압기 본체 - 전주의 온도를 PyQT5를 이용하여 plot
# 아래 변수에 대하여 설정 후 SELECT 할 수 있음
# 1. pole 세 가지 중 하나 선택 가능
# 2. 시작일과 종료일 입력 가능
# 3. 중간 값 채우기 설정 가능
# 4. 3분, 5분, 10분 씩 묶음 기능

# 기능이 동작하는 것은 모두 확인 하였지만 PyQT5를 공부하기 위해 시험삼아 만든 소스이기 때문에 함수화나 구조화는 하지 않았습니다.

#!/usr/bin/env python
# coding: utf-8

import sys

from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QBoxLayout
from PyQt5.QtWidgets import QLineEdit
from PyQt5.QtWidgets import QGroupBox
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QRadioButton
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSlot

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import pandas as pd

import MySQLdb

import random

dict = {'pole':'8132W231', 'time_step':'3T', 'interpolate':'yes', 'time_start':'201611120000', 'time_end':'201611122329'}

class DBCon():

    def __init__(self):
        self.con = MySQLdb.connect('localhost', 'root', '1111', 'kepco')
        self.con.set_character_set('utf8')
        self.cur = self.con.cursor(MySQLdb.cursors.DictCursor)
        self.cur.execute('SET NAMES utf8;')
        self.cur.execute('SET CHARACTER SET utf8;')
        self.cur.execute('SET character_set_connection=utf8;')

    # SELECT 하는 방법
    def select(self, pole_id, time_start, time_end):
        query = """SELECT DATE_FORMAT(CONCAT(CONVERT(COLUMN_MINUTE, CHAR(16)), '00'), '%%Y-%%m-%%d %%H:%%i:%%s') AS COLUMN_MINUTE,POLE_ID,BB_AVG_TEMP,BG_AVG_TEMP,WG_AVG_TEMP,JJ_AVG_TEMP,TH_AVG_TEMP 
                    FROM TB_IOT_POLE_MINUTE_AVG_TEMP 
                    WHERE POLE_ID = '%s' AND COLUMN_MINUTE BETWEEN '%s' AND '%s'  
                    ORDER BY COLUMN_MINUTE""" % (pole_id, time_start, time_end)
        print(query)
        self.cur.execute(query);
        df_query = pd.DataFrame()
        if self.cur.rowcount != 0:
            results = self.cur.fetchall()
            df_query = pd.DataFrame(list(results))
            df_query = df_query.set_index(pd.to_datetime(df_query['COLUMN_MINUTE']))
        print('query_end')
        return df_query

class Form(QWidget):
    def __init__(self):
        QWidget.__init__(self, flags=Qt.Widget)

        # 배치될 위젯 변수 선언
        self.lb_1 = QLabel()
        self.lb_2 = QLabel()
        self.lb_3 = QLabel()
        self.lb_4 = QLabel()
        self.lb_5 = QLabel()

        self.le_time_start = QLineEdit()
        self.le_time_end = QLineEdit()

        # 레이아웃 선언 및 Form Widget에 설정
        # pole 세 가지를 선택할 수 있는 라디오 버튼 설정
        self.grp_layout_pole = QGroupBox("Group Pole")
        self.rb_8132W231 = QRadioButton("8132W231")
        self.rb_8132W212 = QRadioButton("8132W212")
        self.rb_8132W811 = QRadioButton("8132W811")

        # 3분, 5분, 10분 단위로 묶을 수 있는 라디오 버튼 설정
        self.grp_layout_time_step = QGroupBox("Group Time Step")
        self.rb_time_step_3T = QRadioButton("3T")
        self.rb_time_step_5T = QRadioButton("5T")
        self.rb_time_step_10T = QRadioButton("10T")

        # 구간 별로 묶을 때 MIN, MAX, AVG 를 설정하기 위한 라디오 버튼이지만 아직 UI에 구현은 하지 않았음
        self.grp_function = QGroupBox("Group Function")
        self.rb_3_1 = QRadioButton("MIN")
        self.rb_3_2 = QRadioButton("MAX")
        self.rb_3_3 = QRadioButton("AVG")

        # 중간 값을 채울 지 여부 설정
        self.grp_layout_interpolate_flag = QGroupBox("Group Interpolate Flag")
        self.rb_interpolate_yes = QRadioButton("yes")
        self.rb_interpolate_no = QRadioButton("no")

        self.pb_1 = QPushButton()

        # 베이스 레이어, 차트, 폴선택, 시간입력, 중간값설정, 타임스텝설정 레이어를 설정
        self.layout_base = QBoxLayout(QBoxLayout.TopToBottom, self)
        self.layout_screen = QBoxLayout(QBoxLayout.LeftToRight, self)
        self.layout_pole = QBoxLayout(QBoxLayout.LeftToRight, self)
        self.layout_time = QBoxLayout(QBoxLayout.LeftToRight, self)
        self.layout_interpolate_flag = QBoxLayout(QBoxLayout.LeftToRight, self)
        self.layout_time_step = QBoxLayout(QBoxLayout.LeftToRight, self)

        # 폴 세가지를 라디오 버튼으로 추가
        self.layout_pole.addWidget(self.rb_8132W231)
        self.layout_pole.addWidget(self.rb_8132W212)
        self.layout_pole.addWidget(self.rb_8132W811)
        self.grp_layout_pole.setLayout(self.layout_pole)

        # 시간 묶음 3가지를 라디오 버튼으로 추가
        self.layout_time_step.addWidget(self.rb_time_step_3T)
        self.layout_time_step.addWidget(self.rb_time_step_5T)
        self.layout_time_step.addWidget(self.rb_time_step_10T)
        self.grp_layout_time_step.setLayout(self.layout_time_step)

        # 중간값 입력 설정을 라디오 버튼으로 추가
        self.layout_interpolate_flag.addWidget(self.rb_interpolate_yes)
        self.layout_interpolate_flag.addWidget(self.rb_interpolate_no)
        self.grp_layout_interpolate_flag.setLayout(self.layout_interpolate_flag)

        # 실행 버튼 설정(설정에 따라 동작)
        self.button = QPushButton('실 행', self)
        self.button.setToolTip('This is an example button')
        self.button.clicked.connect(self.button_on_click)

        self.layout_base.addLayout(self.layout_screen)
        self.layout_base.addWidget(self.grp_layout_pole)
        self.layout_base.addLayout(self.layout_time)
        self.layout_base.addWidget(self.grp_layout_interpolate_flag)
        self.layout_base.addWidget(self.grp_layout_time_step)
        self.layout_base.addWidget(self.button)
        self.setLayout(self.layout_base)
        self.init_widget()

    def init_widget(self):
        self.setWindowTitle("Layout Basic")
        self.setFixedWidth(1024)
        self.setFixedHeight(768)

        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)
        self.layout_time.addWidget(self.le_time_start)
        self.layout_time.addWidget(self.le_time_end)

        # 폴 별 라디오 버튼 클릭 이벤트
        self.rb_8132W231.toggled.connect(self.rb_8132W231_clicked)
        self.rb_8132W212.toggled.connect(self.rb_8132W2121_clicked)
        self.rb_8132W811.toggled.connect(self.rb_8132W8111_clicked)

        # 시간 묶음 라디오 버튼 클릭 이벤트
        self.rb_time_step_3T.toggled.connect(self.rb_time_step_3T_clicked)
        self.rb_time_step_5T.toggled.connect(self.rb_time_step_5T_clicked)
        self.rb_time_step_10T.toggled.connect(self.rb_time_step_10T_clicked)

        # 중간값 설정 라디오 버튼 클릭 이벤트
        self.rb_interpolate_yes.toggled.connect(self.rb_interpolate_yes_clicked)
        self.rb_interpolate_no.toggled.connect(self.rb_interpolate_no_clicked)
        self.btnClicked()

    def btnClicked(self):
        print('plot_start')

        # 켄버스를 그리기 전에 켄버스를 삭제한다.(삭제하지 않으면 이전 켄버스와 겹쳐 나온다.)
        self.layout_screen.removeWidget(self.canvas)

        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)

        self.layout_screen.addWidget(self.canvas)

        # 사이트 : 1024 x 540
        fig = Figure(figsize=(10.24, 5.4), dpi=100)

        pole_id = dict['pole']
        time_start = dict['time_start']
        time_end = dict['time_end']
        print(dict)
        dbcon = DBCon()
        # DB 에서 SELECT
        df_query = dbcon.select(pole_id, time_start, time_end)
        # SELECT 데이터가 있을 경우 plot
        if df_query.size != 0:
            df_query[['BB_AVG_TEMP','BG_AVG_TEMP','WG_AVG_TEMP','JJ_AVG_TEMP','TH_AVG_TEMP']] = df_query[['BB_AVG_TEMP','BG_AVG_TEMP','WG_AVG_TEMP','JJ_AVG_TEMP','TH_AVG_TEMP']].astype('float64')
            if dict['interpolate'] == '1':
                df_query = df_query[['BB_AVG_TEMP','BG_AVG_TEMP','WG_AVG_TEMP','JJ_AVG_TEMP','TH_AVG_TEMP']].interpolate()
            df_3t = df_query.resample(dict['time_step']).max()
            df_diff = df_3t['BB_AVG_TEMP'] - df_3t['JJ_AVG_TEMP']
            data = [random.random() for i in range(25)]
            # 랜덤 차트
            ax1 = self.fig.add_subplot(211)
            # 변압기 본체, 전주, 변압기 본체 - 전주의 온도를 표시하는 차트
            ax3 = self.fig.add_subplot(212)
            ax1.plot(data, 'r-')
            ax1.set_title('PyQt Matplotlib Example')
            ax3.plot(df_3t['BB_AVG_TEMP'], label='변압기 본체')
            ax3.plot(df_3t['JJ_AVG_TEMP'], label='전주')
            ax3.plot(df_diff, label='차이')
            self.canvas.draw()
        print('plot_end')

    # 폴 클릭 이벤트
    def rb_8132W231_clicked(self, enabled):
        if enabled:
            print('rb_8132W231_clicked')
            dict['pole'] = '8132W231'

    # 폴 클릭 이벤트
    def rb_8132W2121_clicked(self, enabled):
        if enabled:
            print('rb_8132W2121_clicked')
            dict['pole'] = '8132W2121'

    # 폴 클릭 이벤트
    def rb_8132W8111_clicked(self, enabled):
        if enabled:
            print('rb_8132W8111_clicked')
            dict['pole'] = '8132W8111'

    # 시간 묶음 3분 클릭 이벤트
    def rb_time_step_3T_clicked(self, enabled):
        if enabled:
            print('rb_time_step_3T_clicked')
            dict['time_step'] = '3T'

    # 시간 묶음 5분 클릭 이벤트
    def rb_time_step_5T_clicked(self, enabled):
        if enabled:
            print('rb_time_step_5T_clicked')
            dict['time_step'] = '5T'

    # 시간 묶음 10분 클릭 이벤트
    def rb_time_step_10T_clicked(self, enabled):
        if enabled:
            print('rb_time_step_10T_clicked')
            dict['time_step'] = '10T'

    # 중간값 입력 yes 설정 이벤트
    def rb_interpolate_yes_clicked(self, enabled):
        if enabled:
            print('rb_time_step_yes_clicked')
            dict['interpolate'] = '1'

    # 중간값 입력 no 설정 이벤트
    def rb_interpolate_no_clicked(self, enabled):
        if enabled:
            print('rb_time_step_no_clicked')
            dict['interpolate'] = '0'

    # 검색 시작 시간 입력
    def change_dict_time_start(self, str):
        dict['time_start'] = str

    # 검색 종료 시간 입력
    def change_dict_time_end(self, str):
        dict['time_start'] = str

    # 실행 버튼 클릭 이벤트
    def button_on_click(self):
        dict['time_start'] = self.le_time_start.text()
        dict['time_end'] = self.le_time_end.text()
        self.btnClicked()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    form = Form()
    form.show()
    exit(app.exec_())



