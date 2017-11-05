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
        self.grp_layout_pole = QGroupBox("Group Pole")
        self.rb_8132W231 = QRadioButton("8132W231")
        self.rb_8132W212 = QRadioButton("8132W212")
        self.rb_8132W811 = QRadioButton("8132W811")

        self.grp_layout_time_step = QGroupBox("Group Time Step")
        self.rb_time_step_3T = QRadioButton("3T")
        self.rb_time_step_5T = QRadioButton("5T")
        self.rb_time_step_10T = QRadioButton("10T")

        self.grp_function = QGroupBox("Group Function")
        self.rb_3_1 = QRadioButton("MIN")
        self.rb_3_2 = QRadioButton("MAX")
        self.rb_3_3 = QRadioButton("AVG")

        self.grp_layout_interpolate_flag = QGroupBox("Group Interpolate Flag")
        self.rb_interpolate_yes = QRadioButton("yes")
        self.rb_interpolate_no = QRadioButton("no")

        self.pb_1 = QPushButton()

        self.layout_base = QBoxLayout(QBoxLayout.TopToBottom, self)
        self.layout_screen = QBoxLayout(QBoxLayout.LeftToRight, self)
        self.layout_pole = QBoxLayout(QBoxLayout.LeftToRight, self)
        self.layout_time = QBoxLayout(QBoxLayout.LeftToRight, self)
        self.layout_interpolate_flag = QBoxLayout(QBoxLayout.LeftToRight, self)
        self.layout_time_step = QBoxLayout(QBoxLayout.LeftToRight, self)

        self.layout_pole.addWidget(self.rb_8132W231)
        self.layout_pole.addWidget(self.rb_8132W212)
        self.layout_pole.addWidget(self.rb_8132W811)
        self.grp_layout_pole.setLayout(self.layout_pole)

        self.layout_time_step.addWidget(self.rb_time_step_3T)
        self.layout_time_step.addWidget(self.rb_time_step_5T)
        self.layout_time_step.addWidget(self.rb_time_step_10T)
        self.grp_layout_time_step.setLayout(self.layout_time_step)

        self.layout_interpolate_flag.addWidget(self.rb_interpolate_yes)
        self.layout_interpolate_flag.addWidget(self.rb_interpolate_no)
        self.grp_layout_interpolate_flag.setLayout(self.layout_interpolate_flag)

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

        # 레디오 버튼 클릭 이벤트
        self.rb_8132W231.toggled.connect(self.rb_8132W231_clicked)
        self.rb_8132W212.toggled.connect(self.rb_8132W2121_clicked)
        self.rb_8132W811.toggled.connect(self.rb_8132W8111_clicked)

        self.rb_time_step_3T.toggled.connect(self.rb_time_step_3T_clicked)
        self.rb_time_step_5T.toggled.connect(self.rb_time_step_5T_clicked)
        self.rb_time_step_10T.toggled.connect(self.rb_time_step_10T_clicked)

        self.rb_interpolate_yes.toggled.connect(self.rb_interpolate_yes_clicked)
        self.rb_interpolate_no.toggled.connect(self.rb_interpolate_no_clicked)
        self.btnClicked()

    def btnClicked(self):
        print('plot_start')

        self.layout_screen.removeWidget(self.canvas)

        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)

        self.layout_screen.addWidget(self.canvas)

        fig = Figure(figsize=(10.24, 5.4), dpi=100)

        pole_id = dict['pole']
        time_start = dict['time_start']
        time_end = dict['time_end']
        print(dict)
        dbcon = DBCon()
        df_query = dbcon.select(pole_id, time_start, time_end)
        if df_query.size != 0:
            df_query[['BB_AVG_TEMP','BG_AVG_TEMP','WG_AVG_TEMP','JJ_AVG_TEMP','TH_AVG_TEMP']] = df_query[['BB_AVG_TEMP','BG_AVG_TEMP','WG_AVG_TEMP','JJ_AVG_TEMP','TH_AVG_TEMP']].astype('float64')
            if dict['interpolate'] == '1':
                df_query = df_query[['BB_AVG_TEMP','BG_AVG_TEMP','WG_AVG_TEMP','JJ_AVG_TEMP','TH_AVG_TEMP']].interpolate()
            df_3t = df_query.resample(dict['time_step']).max()
            df_diff = df_3t['BB_AVG_TEMP'] - df_3t['JJ_AVG_TEMP']
            data = [random.random() for i in range(25)]
            ax1 = self.fig.add_subplot(211)
            ax3 = self.fig.add_subplot(212)
            ax1.plot(data, 'r-')
            ax1.set_title('PyQt Matplotlib Example')
            ax3.plot(df_3t['BB_AVG_TEMP'], label='변압기 본체')
            ax3.plot(df_3t['JJ_AVG_TEMP'], label='전주')
            ax3.plot(df_diff, label='차이')
            print('plot_start3')
            self.canvas.draw()
        print('plot_end')


    def rb_8132W231_clicked(self, enabled):
        if enabled:
            print('rb_8132W231_clicked')
            dict['pole'] = '8132W231'

    def rb_8132W2121_clicked(self, enabled):
        if enabled:
            print('rb_8132W2121_clicked')
            dict['pole'] = '8132W2121'

    def rb_8132W8111_clicked(self, enabled):
        if enabled:
            print('rb_8132W8111_clicked')
            dict['pole'] = '8132W8111'

    def rb_time_step_3T_clicked(self, enabled):
        if enabled:
            print('rb_time_step_3T_clicked')
            dict['time_step'] = '3T'

    def rb_time_step_5T_clicked(self, enabled):
        if enabled:
            print('rb_time_step_5T_clicked')
            dict['time_step'] = '5T'

    def rb_time_step_10T_clicked(self, enabled):
        if enabled:
            print('rb_time_step_10T_clicked')
            dict['time_step'] = '10T'

    def rb_interpolate_yes_clicked(self, enabled):
        if enabled:
            print('rb_time_step_yes_clicked')
            dict['interpolate'] = '1'

    def rb_interpolate_no_clicked(self, enabled):
        if enabled:
            print('rb_time_step_no_clicked')
            dict['interpolate'] = '0'

    def change_dict_time_start(self, str):
        dict['time_start'] = str

    def change_dict_time_end(self, str):
        dict['time_start'] = str

    @pyqtSlot()
    def button_on_click(self):
        dict['time_start'] = self.le_time_start.text()
        dict['time_end'] = self.le_time_end.text()
        self.btnClicked()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    form = Form()
    form.show()
    exit(app.exec_())



