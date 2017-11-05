import sys
from PyQt5.QtWidgets import QWidget, QApplication, QPushButton, QVBoxLayout, QHBoxLayout
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from datetime import datetime as dt
import pandas_datareader.data as web
import matplotlib.dates as mdates


class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

        self.setLayout(self.layout)
        self.setGeometry(200, 200, 800, 400)

    def initUI(self):

        self.pushButton = QPushButton("DRAW Graph")
        self.pushButton.clicked.connect(self.btnClicked)

        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)

        # btn layout
        btnLayout = QVBoxLayout()
        btnLayout.addWidget(self.canvas)

        # canvas Layout
        canvasLayout = QVBoxLayout()
        canvasLayout.addWidget(self.pushButton)
        canvasLayout.addStretch(1)

        self.layout = QHBoxLayout()
        self.layout.addLayout(btnLayout)
        self.layout.addLayout(canvasLayout)

    def btnClicked(self):
        weeks = mdates.WeekdayLocator(mdates.MONDAY) # x 축 어디에 찍을것인지 지정
        weeksFmt = mdates.DateFormatter('%Y.%m')     # 어떻게 표시할것인지 설정

        sm = web.DataReader('005930.KS', 'yahoo', dt(2017, 1, 1), dt.today())
        sm = sm[sm['Volume'] > 0]

        ax = self.fig.add_subplot(1, 1, 1)
        ax.plot(sm['Adj Close'], label='Adj Close')
        ax.legend(loc='best')
        # ax 에 x 축을 설정해준다
        ax.xaxis.set_major_locator(weeks)
        ax.xaxis.set_major_formatter(weeksFmt)

        ax.grid()
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    app.exec_()
