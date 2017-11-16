import sys
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5.QtGui import QPainter, QColor, QPen
from PyQt5.QtCore import *
from PyQt5 import uic
from tkinter import *

'''
def clickable(widget):

    class Filter(QtWidgets.QGraphicsItem):
        clicked = pyqtSignal()

        def eventFIlter(self, obj, event):
            if obj == widget:
                if event.type() == QEvent.MouseButtonRelease:
                    if obj.rect().contains(event.pos()):
                        self.clicked.emit()
                        return True
            return False

    filter = Filter(widget)
    widget.installEventFilter(filter)
    return filter.clicked


class graphics_Object(QtWidgets.QGraphicsPixmapItem):
    def __init__(self, parent=None):
        super(graphics_Object, self).__init__(parent)
        pixmap = QtGui.QPixmap().scaled(40, 40, QtCore.Qt.KeepAspectRatio)
        self.setPixmap(pixmap)
        self.setFlag(QtGui.QGraphicsPixmapItem.ItemIsSelectable)
        self.setFlag(QtGui.QGraphicsPixmapItem.ItemIsMovable)
        self.setAcceptHoverEvents(True)
        self.switch_LinkedWithItems = []
        self.switch_mininet_name = ''

    def hoverEnterEvent(self, event):
        print('hello')

    def hoverLeaveEvent(self, event):
        print('goodbye')

class graphicsScene(QtWidgets.QGraphicsScene):
    def __init__(self, parent=None):
        super(graphicsScene, self).__init__(parent)

    def mousePressEvent(self, event):
        print('ininininin')
        self.graphics_item = graphics_Object()
        print('ininininin2')

    def mouseReleaseEvent(self, event):
        print('1')
        self.addItem(self.graphics_item.graphics_pixItem)
        print('2')
        self.graphics_item.setPos(event.scenePos())
        print('3')
        self.graphics_item.toggleCircleState()
        print('4')
'''
'''
# 전체적인 Widget을 모두 관리할 수 있음
class controlWidget(QtWidgets.QDialog):
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.ui = uic.loadUi("03_pyQT_GraphicsView.ui", self)
        self.diagramGraphicsView = self.ui.diagramGraphicsView
        self.diagramValueControlTable = self.ui.diagramValueControlTable

    def initWidget(self, scene):
        self.diagramGraphicsView.setScene(scene)
        self.scene = scene

    def getUi(self):
        return self.ui

    def getScene(self):
        return self.scene

    def initTableShape(self, rowCnt, colCnt):
        self.diagramValueControlTable.setRowConut(6)
        self.diagramValueControlTable.setColumnCount(3)
        print('dd')
'''


# GraphicsScene Event용 class
class DiagramScene(QtWidgets.QGraphicsScene):
    itemSelected = pyqtSignal(QtWidgets.QGraphicsItem)

    def __init__(self, parent=None):
        super(DiagramScene, self).__init__(parent)

    def mousePressEvent(self, mouseEvent):
        if (mouseEvent.button() != Qt.LeftButton):
            return

        selectedItems = self.items(mouseEvent.scenePos())  # 선택된 객체 목록을 리스트 형태로 반환
        selectedItem = selectedItems[0]  # 선택된 항목 리스트 중 첫번째 객체 획득
        # selectedItem.getCircleLocation()
        selectedItem.toggleState()
        super(DiagramScene, self).mousePressEvent(mouseEvent)


# 원 그리기 class
class drawCircleWidget(QtWidgets.QDialog):
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.ui = uic.loadUi("03_pyQT_GraphicsView.ui", self)
        self.diagramGraphicsView = self.ui.diagramGraphicsView
        self.brush = QtGui.QBrush(QColor(Qt.red))
        self.pen = QtGui.QPen(QColor(Qt.red))
        self.circle = None

    def setUi(self, scene):
        self.scene = scene
        self.diagramGraphicsView.setScene(scene)
        self.setMouseTracking(True)

    def initiWidget(self, parentObj):
        self.ui = parentObj.getUi()
        self.diagramGraphicsView = self.ui.diagramGraphicsView
        self.scene = parentObj.getScene()
        self.setMouseTracking(True)

    def createCircleObj(self, posX, posY, radX, radY):
        self.circle = circleGraphicObject(posX, posY, radX, radY)
        self.circle.setBrush(self.brush)
        self.scene.addItem(self.circle)


# Ellipse 도형 관리용 class
class circleGraphicObject(QtWidgets.QGraphicsEllipseItem):
    def __init__(self, posX, posY, radX, radY):
        self.posX, self.posY, self.radX, self.radY = posX, posY, radX, radY
        super(circleGraphicObject, self).__init__(posX, posY, radX, radY)

    def getCircleLocation(self):
        print("Location : ", self.posX, " , ", self.posY, " , ", self.radX, " , ", self.radY)

    # 원의 상태를 바꿔준다.
    def toggleState(self):
        tmp_brush = QtGui.QBrush(QColor(Qt.white))
        tmp_pen = QtGui.QPen(QColor(Qt.white))

        if self.brush().color() == QColor(Qt.red):
            tmp_brush = QtGui.QBrush(QColor(Qt.black))
            tmp_pen = QtGui.QPen(QColor(Qt.black))
        else:
            tmp_brush = QtGui.QBrush(QColor(Qt.red))
            tmp_pen = QtGui.QPen(QColor(Qt.red))

        self.setBrush(tmp_brush)
        self.setPen(tmp_pen)


'''
    def mousePressEvent(self, event):
        print("click")

        print("ee " + event.type())

        self.offset = event.pos()
        selectedItem = event.widget

        QtGui.QWidget.mousePressEvent(self, event)
        print(selectedItem)


        self.toggleCircleState()
        # QtGui.QWidget.mousePressEvent(self, event)

    if source is self.circle:
        if event.type() == QtCore.QEvent.MouseButtonPress:
             self.toggleCircleState()

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

# App Part
# app = QtWidgets.QApplication(sys.argv)
app = QtWidgets.QApplication([])
# root = Tk()

scene = DiagramScene()

# mainControlObj = controlWidget()
# mainControlObj.initWidget(scene)

# scene = QtWidgets.QGraphicsScene()
# scene = graphicsScene()
# scene.bind("<Button-1>", mouseClieckedWidgetEvent)

# layout = QtWidgets.QHBoxLayout()
# layout.addStretch(1)

circles = drawCircleWidget()
circles.setUi(scene)
circles.createCircleObj(-110, 0, 50, 50)
# circles.circle.getCircleLocation()
# circles.bind("<Button-1>", onClick)

circles2 = drawCircleWidget()
circles2.setUi(scene)
circles2.createCircleObj(0, 0, 50, 50)
# circles2.bind("<Button-1>", onClick)

circles3 = drawCircleWidget()
circles3.setUi(scene)
circles3.createCircleObj(110, 0, 50, 50)
# circles3.bind("<Button-1>", onClick)

circles4 = drawCircleWidget()
circles4.setUi(scene)
circles4.createCircleObj(0, 100, 50, 50)

circles.show()
# circles2.show()
# circles3.show()
# circles4.show()
sys.exit(app.exec())
# app.exec_()





