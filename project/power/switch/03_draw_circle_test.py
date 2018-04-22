# coding: utf-8
import sys
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5.QtGui import QPainter, QColor, QPen, QTransform
from PyQt5.QtCore import *
from PyQt5 import uic
from tkinter import *
from PyQt5.QtGui import QMouseEvent

import traceback

class drawCircleBtn(QtWidgets.QDialog):

    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.ui = uic.loadUi("03_pyQT_GraphicsView.ui", self)
        self.brush = QtGui.QBrush(QColor(Qt.red))
        self.pen = QtGui.QPen(QColor(Qt.red))
        self.circle = None


    def setUi(self, scene):
        self.ui.graphicsView.setScene(scene)
        self.scene = scene

    def createCircleObj(self, posX, posY, radX, radY):
        self.circle = QtWidgets.QGraphicsEllipseItem(posX, posY, radX, radY)
        self.circle.setBrush(self.brush)
        self.scene.addItem(self.circle)
        # self.scene.circle.clicked.mouseReleaseEvent(self.mousePressEvent)

    # 원의 상태를 바꿔준다.
    def toggleCircleState(self):
        tmp_brush = QtGui.QBrush(QColor(Qt.white))
        tmp_pen = QtGui.QPen(QColor(Qt.white))

        if self.circle.brush().color() == QColor(Qt.red):
            tmp_brush = QtGui.QBrush(QColor(Qt.black))
            tmp_pen = QtGui.QPen(QColor(Qt.black))
        else:
            tmp_brush = QtGui.QBrush(QColor(Qt.red))
            tmp_pen = QtGui.QPen(QColor(Qt.red))

        self.circle.setBrush(tmp_brush)
        self.circle.setPen(tmp_pen)

    def mousePressEvent(self, event):
        try:
            print("click")
            print("event_type", event.type())
            print('event_button:',event.button())
            #self.offset = event.pos()
            print('x:',event.pos().x())
            print('y:',event.pos().y())
            self.toggleCircleState()

        except Exception as e:
            print('err:Exception')
            traceback.print_exc()



    '''
    if source is self.circle:
        if event.type() == QtCore.QEvent.MouseButtonPress:
             self.toggleCircleState()
    '''


'''
    def paintEvent(self, event):
        paint = QtGui.QPainter()
        paint.begin(self)
        # optional
        paint.setRenderHint(QPainter.Antialiasing)
        # make a white drawing background
        paint.setBrush(Qt.white)
        paint.drawRect(event.rect())
        # 원의 지름 : 원으로 만들기 위해 값을 동일하게 지정
        # x < y 이면 원의 모양이 0 처럼 생김
        radx = 10  # 가로 지름
        rady = 10  # 세로 지름
        # 원의 테두리색
        paint.setPen(Qt.red)
        # 원 그리기
        center = QPoint(125, 125)
        # 원을 채울 색상으로 brush 세팅
        paint.setBrush(Qt.black)
        paint.drawEllipse(center, radx, rady)



        center = QPoint(145, 145)
        paint.setBrush(Qt.black)
        temp = paint.drawEllipse(center, radx, rady)
        print(type(temp))
        print(temp)
        
        for i in range(125, 220, 10):
            center = QPoint(i, i)
            # optionally fill each circle black
            paint.setBrush(Qt.black)
            paint.drawEllipse(center, radx, rady)
        
        paint.end()
'''


# App
# app = QtWidgets.QApplication(sys.argv)
app = QtWidgets.QApplication([])
root = Tk()

scene = QtWidgets.QGraphicsScene()
# scene.bind("<Button-1>", mouseClieckedWidgetEvent)

circles = drawCircleBtn()
circles.setUi(scene)
circles.createCircleObj(0, 0, 100, 100)
# circles.bind("<Button-1>", onClick)

circles2 = drawCircleBtn()
circles2.setUi(scene)
circles2.createCircleObj(110, 0, 100, 100)
# circles2.bind("<Button-1>", onClick)

circles3 = drawCircleBtn()
circles3.setUi(scene)
circles3.createCircleObj(-110, 0, 100, 100)
# circles3.bind("<Button-1>", onClick)

circles2.show()
sys.exit(app.exec())
# app.exec_()





