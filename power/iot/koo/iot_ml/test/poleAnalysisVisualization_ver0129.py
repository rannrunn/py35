import sys
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush
from PyQt5.QtCore import *
from PyQt5 import uic
from PyQt5.QtWidgets import QBoxLayout
from PyQt5.QtWidgets import QApplication, QMainWindow, QSizePolicy, QPushButton

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import random


# GraphicsScene Event용 class
class DiagramScene(QtWidgets.QGraphicsScene):
    # itemSelected = pyqtSignal(QtWidgets.QGraphicsItem)
    # drawTemperatureGraphCanvas = None
    def __init__(self, parent=None):
        super(DiagramScene, self).__init__(parent)
        # DiagramScene.drawTemperatureGraphCanvas = PlotCanvas(self, width=5, height=2)
        # drawTemperatureGraphCanvas.drawTemperatureGraph(resultTemperatureData)

    def mousePressEvent(self, mouseEvent):
        if (mouseEvent.button() != Qt.LeftButton):
            return

        selectPoint, selectPoleObject = mouseEvent.scenePos().toPoint(), None
        poleWidth, poleHeight = PoleObject.getShortLineLength(), PoleObject.getCenterLineLength()

        print(selectPoint)
        for poleIdItem in ObjectControlClass.poleObjectDic.keys():
            tmpPoleObject = ObjectControlClass.poleObjectDic[poleIdItem]
            poleObjectStPoint = tmpPoleObject.getLocationAsPointObject()
            poleObjectDestPoint = QPoint(poleObjectStPoint.x() + poleWidth, poleObjectStPoint.y() + poleHeight)
            poleRangeRect = QRect(poleObjectStPoint, poleObjectDestPoint)

            # print('poleRangeRect.contains(selectPoint)', poleRangeRect.contains(selectPoint))
            if poleRangeRect.contains(selectPoint):
                selectPoleObject = tmpPoleObject
                # print(selectPoleObject.getPoleID())
                break

        if selectPoleObject is None:
            print('selectPoleObject is None')
            return

        ObjectControlClass.tmpClassObj.displayTemperatureGraph(int(ObjectControlClass.selectDisplayMode), selectPoleObject.getPoleID())

        # print(mouseEvent.scenePos())
        # selectedItems = self.items(mouseEvent.scenePos())  # 선택된 객체 목록을 리스트 형태로 반환
        #  # 선택된 항목 리스트 중 첫번째 객체 획득
        # # selectedItem.getCircleLocation()
        # # if type(selectedItem) == QtWidgets.QGraphicsEllipseItem.type():
        # # print(type(selectedItem))
        # # print(circleGraphicObject)
        # if selectedItems.__len__() > 0:
        #     selectedItem = selectedItems[0]
        #
        #     # print(type(selectedItem))
        #     # print(type(selectedItem.parentItem()))
        #     #
        #     # if selectedItem.getLocation() is not None:
        #     #     print('selectedItem.getLocation() is not None')
        #
        #     # QLabel Object 선택 시 return 처리
        #     # if type(selectedItem) == QtWidgets.QGraphicsProxyWidget:
        #     #     # print(type(selectedItem))
        #     #     # print('label in')
        #     #     return
        #     # elif type(selectedItem) == QtWidgets.QGraphicsSimpleTextItem:
        #     #     # print('in')
        #     #     print('click widget info : ')
        #     #     selectedItem.parentItem().getLocation()
        #     #     selectedItem.parentItem().toggleState()
        #     #     # print(type(selectedItem.parentItem()))
        #     #     return
        #     #
        #     # selectedItem.toggleState()
        #     # selectedItem.testingFunc()
        #     super(DiagramScene, self).mousePressEvent(mouseEvent)
        #
        #     # if type(selectedItem) == circleGraphicObject:
        #     #     print('in circle object click event')
        #     #     selectedItem.toggleState()
        #     #     super(DiagramScene, self).mousePressEvent(mouseEvent)
        #     # elif type(selectedItem) == rectGraphicObject:
        #     #     print('in rect object click event')
        #     # elif type(selectedItem) == lineGraphicObject:
        #     #     print('in line object click event')
        #     #     selectedItem.toggleState()
        #     #     super(DiagramScene, self).mousePressEvent(mouseEvent)
        # else:
        #     # print('else object click')
        #     return

        super(DiagramScene, self).mousePressEvent(mouseEvent)

class ObjectControlClass(QtWidgets.QDialog):
    diagramScene = None
    dialogVerticalLayout = None
    dialogScreenLayout = None
    unbalanceDataTable = None
    gapOfPole = None
    poleObjectDic = {}
    tableIndexOfPoleObject = {}
    betweenObjectMarginValue = {}
    displayModeValue = {}
    lineDictionary = {}
    LastIndex = -1
    tmpClassObj = None
    selectDisplayMode = 0

    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.ui = uic.loadUi("IoTPoleGraphicsView.ui", self)
        self.dialogHorizontalLayout = self.ui.dialogHorizontalLayout
        self.dialogVerticalLayout = self.ui.dialogVerticalLayout

        self.unbalanceDataTable = None      # 불균형 정보를 저장할 테이블

        ObjectControlClass.tmpClassObj = tmpClass()

        self.setBetweenObjectMarginValue()
        self.setDisplayModeValue()
        self.setUi()

    def getDialogHorizontalLayout(self):
        return self.dialogHorizontalLayout

    def getDialogVerticalLayout(self):
        return self.dialogVerticalLayout

    def setBetweenObjectMarginValue(self):

        topValue = [0, -300]
        leftValue = [-150, 0]
        rightValue = [150, 0]
        bottomValue = [0, 300]

        ObjectControlClass.betweenObjectMarginValue['Top'] = topValue
        ObjectControlClass.betweenObjectMarginValue['Bottom'] = bottomValue
        ObjectControlClass.betweenObjectMarginValue['Left'] = leftValue
        ObjectControlClass.betweenObjectMarginValue['Right'] = rightValue

    def setDisplayModeValue(self):
        ObjectControlClass.displayModeValue['Total'] = 0
        ObjectControlClass.displayModeValue['Monthly'] = 1
        ObjectControlClass.displayModeValue['Daily'] = 2

    def setUi(self):
        # diagramScene = DiagramScene()
        #
        # tableWidget = QtWidgets.QTableWidget()
        # self.scene = DiagramScene()
        # self.diagramGraphicsView.setScene(scene)

        # if ObjectControlClass.dialogScreenLayout is None:
        #     ObjectControlClass.dialogScreenLayout = QBoxLayout(QBoxLayout.LeftToRight, self)
        #     ObjectControlClass.dialogVerticalLayout.addLayout(ObjectControlClass.dialogScreenLayout)


        if ObjectControlClass.diagramScene is None:
            ObjectControlClass.diagramScene = DiagramScene()
            # ObjectControlClass.diagramScene.setParent(ObjectControlClass.dialogVerticalLayout)
            # ObjectControlClass.dialogVerticalLayout.addWidget(ObjectControlClass.diagramScene)

        self.diagramGraphicsView.setScene(ObjectControlClass.diagramScene)
        self.setMouseTracking(True)

        if ObjectControlClass.unbalanceDataTable == None:
            ObjectControlClass.unbalanceDataTable = QtWidgets.QTableWidget()
            self.setTableInfo()
            # self.setTableWidgetObject()
            # self.initDiagramTableWidget()
            # ObjectControlClass.diagramTableWidget.itemChanged.connect(self.tableItemChangedEvent)
            # ObjectControlClass.diagramTableWidget.itemClicked.connect(self.tableItemClickedEvent)
            # ObjectControlClass.unbalanceDataTable.itemChanged.connect(self.tableItemChangedEvent)
            ObjectControlClass.unbalanceDataTable.itemClicked.connect(self.tableItemClickedEvent)
            ObjectControlClass.unbalanceDataTable.setMinimumWidth(350)
            ObjectControlClass.unbalanceDataTable.setMaximumWidth(350)
            ObjectControlClass.unbalanceDataTable.verticalHeader().setVisible(False)

    def setTableInfo(self):
        self.setTableWidgetObject()                 # 테이블을 Graphic Scene에 등록
        self.initDiagramTableWidget()               # Table의 스펙(헤더의 수와 값, 크기) 지정

    def setTableWidgetObject(self):
        self.dialogHorizontalLayout.addWidget(ObjectControlClass.unbalanceDataTable)
        self.setLayout(self.dialogHorizontalLayout)

    def initDiagramTableWidget(self):
        ObjectControlClass.unbalanceDataTable.resize(10, 30)
        ObjectControlClass.unbalanceDataTable.setColumnCount(3)
        ObjectControlClass.unbalanceDataTable.setRowCount(0)
        ObjectControlClass.unbalanceDataTable.setHorizontalHeaderLabels(['Pole Id', 'unbalanceClass', 'unbalanceInfo'])

    def tableItemClickedEvent(self):
        selectedItems = ObjectControlClass.unbalanceDataTable.selectedItems()
        if selectedItems.__len__() > 0:
            # print('selectedItems.__len__() : ', selectedItems.__len__())
            selectedItem = selectedItems[0]
            selectedRowNum = selectedItem.row()
            # print("row : ", selectedItem.row())

            # print('selectedRowNum : ', selectedRowNum)
            # print(drawWidget.caseCount)
            if selectedRowNum > 2:
                return

            ObjectControlClass.selectDisplayMode = selectedRowNum
            ObjectControlClass.tmpClassObj.calcPoleUnbalance(int(selectedRowNum), ObjectControlClass.tableIndexOfPoleObject.values())

    def insertDisplayModeValueIntoTable(self):
        # print('DIc Value : ', ObjectControlClass.displayModeValue['Total'])
        # print('DIc Value int : ', int(ObjectControlClass.displayModeValue['Total']))
        ObjectControlClass.insertUnbalanceDataIntoTable(int(ObjectControlClass.displayModeValue['Total']), '누적', '-', '-')
        ObjectControlClass.insertUnbalanceDataIntoTable(int(ObjectControlClass.displayModeValue['Monthly']), '월별', '-', '-')
        ObjectControlClass.insertUnbalanceDataIntoTable(int(ObjectControlClass.displayModeValue['Daily']), '일별', '-', '-')

    # 객체 생성을 위해 별도로 호출하는 함수
    def createPoleObject(self, poleID):
        self.poleObject = PoleObject(poleID)
        tmpPointX, tmpPointY = self.calcStandardPoint()
        self.poleObject.setLocation(tmpPointX, tmpPointY)
        self.poleObject.drawPoleWidget()

        if ObjectControlClass.LastIndex == -1:
            self.insertDisplayModeValueIntoTable()

        self.insertPoleObjectIntoDictionary()
        self.poleObject.addPoleWidget(ObjectControlClass.diagramScene)

        if ObjectControlClass.LastIndex != 0:
            self.drawLInkPoleLine()

        # ObjectControlClass.diagramScene.addWidget(self.poleObject)

    def calcStandardPoint(self):
        tmpPointX, tmpPointY = 0, 0
        print(len(self.poleObjectDic))
        if len(self.poleObjectDic) == 0:
            tmpPointX = 250
            tmpPointY = 150
        else:
            parentIndex = (int(ObjectControlClass.LastIndex / 3) - 1) * 3 + 1
            print('parentIndex : ', parentIndex)
            if parentIndex < 0:
                parentIndex = 0
            print('parentIndex : ', parentIndex)
            # print('parentIdex : ', parentIdex)
            # print('LastIndex + 1  : ', ObjectControlClass.LastIndex + 1)
            parentPoleObject = self.getParentPoleUsingPoleIndex(parentIndex)
            # parentPoleId = ObjectControlClass.tableIndexOfPoleObject[parentIdex]
            # parentPoleObject = ObjectControlClass.poleObjectDic[parentPoleId]
            parentPointX, parentPointY = parentPoleObject.getLocation()
            # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            poleDirectionValue = self.calcPoleDirection()
            self.setPoleRelation(parentPoleObject, poleDirectionValue)
            # self.setPoleDirection()
            gapLocation = ObjectControlClass.betweenObjectMarginValue[self.poleObject.getPoleDirection()]
            tmpPointX = parentPointX + gapLocation[0]
            tmpPointY = parentPointY + gapLocation[1]

            print('Currunt Pole Point : ( ', tmpPointX, ', ', tmpPointY, ')')

        return tmpPointX, tmpPointY

    def getParentPoleUsingPoleIndex(self, parentIndex):
        print('parent Index : ', parentIndex)
        # print('parentPoleId : ', parentPoleId)
        print('tableIndexOfPoleObject length : ', len(ObjectControlClass.tableIndexOfPoleObject.keys()))
        parentPoleId = ObjectControlClass.tableIndexOfPoleObject[parentIndex]

        parentPoleObject = ObjectControlClass.poleObjectDic[parentPoleId]
        return parentPoleObject

    def setPoleRelation(self, parentPole, poleDirection):
        parentPole.setChildPole(self.poleObject, poleDirection)
        self.poleObject.setParentPole(parentPole)
        self.poleObject.setPoleDirection(poleDirection)

    def calcPoleDirection(self):
        chkIndexValue = (ObjectControlClass.LastIndex + 1) % 3
        tmpDirection = None

        print('LastIndex +1 : ', ObjectControlClass.LastIndex + 1)
        print('LastIndex +1 %3: ', (ObjectControlClass.LastIndex + 1) % 3)
        print('chkIndexValue: ', chkIndexValue)
        if chkIndexValue == 0:
            tmpDirection = 'Bottom'
        elif chkIndexValue == 1:
            tmpDirection = 'Right'
        else:
            tmpDirection = 'Top'

        print('tmpDirection : ', tmpDirection)

        return tmpDirection

    # def getPoleDirection(self):
    #     return self.poleDirection

    def insertPoleObjectIntoDictionary(self):
        ObjectControlClass.LastIndex += 1
        ObjectControlClass.tableIndexOfPoleObject[ObjectControlClass.LastIndex] = self.poleObject.getPoleID()
        ObjectControlClass.poleObjectDic[self.poleObject.getPoleID()] = self.poleObject

    def drawLInkPoleLine(self):
        poleDirection = self.poleObject.getPoleDirection()
        parentPolePointX, parentPolePointY = self.poleObject.getParentPole().getLocation()
        linkLineToPointX, linkLineToPointY, linkLineFromPointX, linkLineFromPointY  = None, None, None, None
        LengthOfParentPoleHeight, LengthOfParentPoleWidth = PoleObject.getCenterLineLength(), PoleObject.getShortLineLength()
        centerLinePenWidth = self.poleObject.getCenterLinePenWidth()

        gapOfPointX, gapOfPointY = ObjectControlClass.betweenObjectMarginValue[poleDirection]

        if poleDirection == 'Bottom':
            linkLineFromPointX = parentPolePointX + int(LengthOfParentPoleWidth / 2)
            linkLineFromPointY = parentPolePointY + LengthOfParentPoleHeight + int(centerLinePenWidth / 2)

            linkLineToPointX = parentPolePointX + int(LengthOfParentPoleWidth / 2)
            linkLineToPointY = parentPolePointY + gapOfPointY - int(centerLinePenWidth / 2)

        elif poleDirection == 'Right':
            linkLineFromPointX = parentPolePointX + int(LengthOfParentPoleWidth / 2) + int(centerLinePenWidth / 2)
            linkLineFromPointY = parentPolePointY + int(LengthOfParentPoleHeight / 2)

            linkLineToPointX = parentPolePointX + gapOfPointX + int(LengthOfParentPoleWidth / 2) - int(centerLinePenWidth / 2)
            linkLineToPointY = parentPolePointY + int(LengthOfParentPoleHeight / 2)
        elif poleDirection == 'Top':
            linkLineFromPointX = parentPolePointX + int(LengthOfParentPoleWidth / 2)
            linkLineFromPointY = parentPolePointY - int(centerLinePenWidth / 2)

            linkLineToPointX = parentPolePointX + int(LengthOfParentPoleWidth / 2)
            linkLineToPointY = parentPolePointY + gapOfPointY + LengthOfParentPoleHeight + int(centerLinePenWidth / 2)

        print('linkLinePoint : ( ', linkLineToPointX, ', ', linkLineToPointY, ' ) ~ ( ', linkLineFromPointX, ', ', linkLineFromPointY, ' )')

        linkedLineObject = QtWidgets.QGraphicsLineItem(linkLineFromPointX, linkLineFromPointY, linkLineToPointX, linkLineToPointY)
        linkedLineObject.setPen(QPen(QColor(Qt.cyan), 2, Qt.SolidLine, Qt.FlatCap, Qt.RoundJoin))
        self.poleObject.diagramScene.addItem(linkedLineObject)
        self.poleObject.getParentPole().setLinkedChildLine(linkedLineObject, poleDirection)
    #     setLinkedChildLine

    #     ObjectControlClass.diagramScene
#     self.calcLinePoint()
#     self.centerLineObject = QLineF(self.centerX, self.centerY, self.centerX, self.centerY + PoleObject.centerLineLength)
#     self.shortFirstLineObject = QLineF(self.firstLineX, self.firstLineY, self.firstLineX + PoleObject.shortLineLength, self.firstLineY)
#     self.shortSecondLineObject = QLineF(self.secondLineX, self.secondLineY, self.secondLineX + PoleObject.shortLineLength, self.secondLineY)
#
#     self.setLine(self.centerLineObject)
#     self.setLine(self.shortFirstLineObject)
#     self.setLine(self.shortSecondLineObject)
#
#
# def addPoleWidget(self, diagramScene):
#     self.diagramScene = diagramScene
#     self.drawCenterLinePen = QtGui.QPen(QColor(Qt.darkGray))
#     self.drawCenterLinePen.setWidth(10)
#
#     self.drawShortLinePen = QtGui.QPen(QColor(Qt.darkGray))
#     self.drawShortLinePen.setWidth(8)
#     diagramScene.addLine(self.centerLineObject, self.drawCenterLinePen)
#     diagramScene.addLine(self.shortFirstLineObject, self.drawShortLinePen)
#     diagramScene.addLine(self.shortSecondLineObject, self.drawShortLinePen)

    # unbalanceDataTable에 데이터를 넣어주는 부분
    # Pole 객체 생성 시에 함께 넣어준다.
    def insertUnbalanceDataIntoTable(idx, poleID, unbalanceClass, unbalanceInfo):
        ObjectControlClass.unbalanceDataTable.insertRow(idx)
        displayUnbalceClassValue = ObjectControlClass.makeDisplayUnbalceClassValue(unbalanceClass)
        ObjectControlClass.unbalanceDataTable.setItem(idx, 0, QtWidgets.QTableWidgetItem(str(poleID)))
        ObjectControlClass.unbalanceDataTable.setItem(idx, 1, QtWidgets.QTableWidgetItem(str(displayUnbalceClassValue)))
        ObjectControlClass.unbalanceDataTable.setItem(idx, 2, QtWidgets.QTableWidgetItem(str(unbalanceInfo)))

    def makeDisplayUnbalceClassValue(unbalanceClass):
        resultData = '위험군 분류값 없음'
        if unbalanceClass == 'High':
            resultData = '고위험군'
        elif unbalanceClass == 'Low':
            resultData = '저위험군'
        else:
            resultData = '일반'

        return resultData

class PoleObject(QtWidgets.QGraphicsLineItem, QtWidgets.QGraphicsEllipseItem):
    def __init__(self, poleID):
        super(PoleObject, self).__init__()

        self.poleID = poleID
        self.unbalanceClass = None          # 고위험군, 저위험군, 일반
        self.unbalanceInfo = None           # 불균형 횟수 데이터

        self.centerLineObject = None
        self.shortFirstLineObject = None
        self.shortSecondLineObject = None
        self.unbalanceStateCircle = None

        self.parentPole = None
        self.childPoles = {}
        self.poleDirection = None                   # 부모에 대해 상대적인 위치 정보를 저장\
        self.linkedChildLine = {}
        self.centerLinePenWidth = 10
        self.shortLinePenWidth = 8

        PoleObject.unbalanceClassValue = {}
        PoleObject.unbalanceInfoValue = {}
        PoleObject.unbalanceInfoSize = {}
        PoleObject.unbalanceColor = {}

        # PoleObject.nodeDirectionValue = ['Right', 'Top', 'Bottom']
        PoleObject.setClassificationValue()
        self.setPoleDesign()

    def setPoleDesign(self):
        PoleObject.centerLineLength = 180
        PoleObject.shortLineLength = 50

        PoleObject.firstShortLineY = 30
        PoleObject.gapOfShortLineY = 15

        # 원을 그릴 때 사용할 원의 지름을 지정
        PoleObject.unbalanceInfoSize['Big'] = 35
        PoleObject.unbalanceInfoSize['Medium'] = 25
        PoleObject.unbalanceInfoSize['Small'] = 15


        # 불균형 위험군 분류에 따른 원의 색상 지정
        PoleObject.unbalanceColor['High'] = QColor(QColor(255, 0, 0, 100))
        # PoleObject.unbalanceColor['Low'] = QtGui.QBrush(QColor(Qt.yellow))
        PoleObject.unbalanceColor['Low'] = QColor(QColor(253, 126, 23, 100))
        PoleObject.unbalanceColor['Normal'] = QColor(QColor(0, 255, 0, 100))

        self.drawCenterLinePen = QtGui.QPen(QColor(Qt.darkGray))
        self.drawCenterLinePen.setWidth(self.centerLinePenWidth)

        self.drawShortLinePen = QtGui.QPen(QColor(Qt.darkGray))
        self.drawShortLinePen.setWidth(self.shortLinePenWidth)

    def displayPoleInfo(self):
        print('************ Pole Location Info ************')
        print(' # Pole ID : ', self.poleID)
        print(' # Pole Point : ( ', self.pointX, ', ', self.pointY, ' )')
        print(' # Unbalance Class : ', self.unbalanceClass)
        print(' # Unbalance Info : ', self.unbalanceInfo)
        print()

    def getLocationAsPointObject(self):
        self.displayPoleInfo()
        return self.pointObject

    def setLocationAsPointObject(self, pointObject):
        self.pointObject = pointObject
        # self.setLocation(self.pointF.x(), self.pointF.y())
        self.displayPoleInfo()

    def getLocation(self):
        self.displayPoleInfo()
        return self.pointX, self.pointY

    def setLocation(self, pointX, pointY):
        self.pointX, self.pointY = pointX, pointY
        self.setLocationAsPointObject(QPoint(self.pointX, self.pointY))
        self.displayPoleInfo()

    def getCenterLineLength():
        return PoleObject.centerLineLength

    def getShortLineLength():
        return PoleObject.shortLineLength

    def setClassificationValue():
        PoleObject.unbalanceClassValue['High'] = '0'
        PoleObject.unbalanceClassValue['Low'] = '1'
        PoleObject.unbalanceClassValue['Normal'] = '2'

        PoleObject.unbalanceInfoValue['Big'] = 60
        PoleObject.unbalanceInfoValue['Medium'] = 30
        # PoleObject.unbalanceInfoValue['Small'] = 30

    def getPoleInfo(self):
        self.displayPoleInfo()
        return self.poleID, self.unbalanceClass, self.unbalanceInfo

    def setPoleInfo(self, unbalanceClass, unbalanceInfo):
        self.unbalanceClass, self.unbalanceInfo = unbalanceClass, unbalanceInfo
        self.displayPoleInfo()

    def getUnbalanceClass(self):
        return self.unbalanceClass

    def setUnbalanceClass(self, unbalanceClass):
        self.unbalanceClass = unbalanceClass

    def getUnbalanceInfo(self):
        return self.unbalanceInfo

    def getParentPole(self):
        return self.parentPole

    def setParentPole(self, parentPoleItem):
        self.parentPole = parentPoleItem

    def getChildPole(self, childDirection):
        return self.childPoles[childDirection]

    def setChildPole(self, childPoleItem, childPoleDirection):
        self.childPoles[childPoleDirection] = childPoleItem

    def getPoleDirection(self):
        return self.poleDirection

    def setPoleDirection(self, poleDirection):
        self.poleDirection = poleDirection

    def getLinkedChildLine(self, linkedChildDirection):
        return self.linkedChildLine[linkedChildDirection]

    def setLinkedChildLine(self, linkedChildLineObject, linkedChildDirection):
        self.linkedChildLine[linkedChildDirection] = linkedChildLineObject

    def setUnbalanceInfo(self, unbalanceInfo):
        self.unbalanceInfo = unbalanceInfo
        self.setUnbalanceInfoSizeValue()

    def setUnbalanceInfoSizeValue(self):
        resultData = None
        if self.unbalanceInfo >= PoleObject.unbalanceInfoSize['Big']:
            resultData = 'Big'
        elif self.unbalanceInfo >= PoleObject.unbalanceInfoSize['Medium']:
            resultData = 'Medium'
        else:
            resultData = 'Small'

        self.unbalanceInfoSizeValue = resultData

    def getPoleID(self):
        return self.poleID

    def setPoleID(self, poleID):
        self.poleID = poleID

    def calcLinePoint(self):
        self.centerX, self.centerY = self.pointX + PoleObject.shortLineLength / 2, self.pointY
        self.firstLineX, self.firstLineY = self.pointX, self.pointY + PoleObject.firstShortLineY
        self.secondLineX, self.secondLineY = self.pointX, self.firstLineY + PoleObject.gapOfShortLineY

    def drawPoleWidget(self):
        self.calcLinePoint()
        self.centerLineObject = QLineF(self.centerX, self.centerY, self.centerX, self.centerY + PoleObject.centerLineLength)
        self.shortFirstLineObject = QLineF(self.firstLineX, self.firstLineY, self.firstLineX + PoleObject.shortLineLength, self.firstLineY)
        self.shortSecondLineObject = QLineF(self.secondLineX, self.secondLineY,  self.secondLineX + PoleObject.shortLineLength, self.secondLineY)

        self.setLine(self.centerLineObject)
        self.setLine(self.shortFirstLineObject)
        self.setLine(self.shortSecondLineObject)

        # self.centerLineObject.setParentItem(self)
        # self.centerLineObject.setParentItem(self)
        # self.centerLineObject.setParentItem(self)

    def addPoleWidget(self, diagramScene):
        self.diagramScene = diagramScene
        self.drawCenterLinePen = QtGui.QPen(QColor(Qt.darkGray))
        self.drawCenterLinePen.setWidth(10)

        self.drawShortLinePen = QtGui.QPen(QColor(Qt.darkGray))
        self.drawShortLinePen.setWidth(8)
        diagramScene.addLine(self.centerLineObject, self.drawCenterLinePen)
        diagramScene.addLine(self.shortFirstLineObject, self.drawShortLinePen)
        diagramScene.addLine(self.shortSecondLineObject, self.drawShortLinePen)

    def drawUnbalanceStateCircle(self):
        print('self.unbalanceInfoSizeValue : ', self.unbalanceInfoSizeValue)
        self.circleWidth, self.circleHeight = PoleObject.unbalanceInfoSize[self.unbalanceInfoSizeValue], PoleObject.unbalanceInfoSize[self.unbalanceInfoSizeValue]
        self.circleCenterPointX, self.circleCenterPointY = self.firstLineX + (PoleObject.shortLineLength / 2), self.firstLineY + (PoleObject.gapOfShortLineY / 2)
        self.circlePointX, self.circlePointY = self.circleCenterPointX - (self.circleWidth / 2), self.circleCenterPointY - (self.circleHeight / 2)
        self.unbalanceStateCircle = QtWidgets.QGraphicsEllipseItem(self.circlePointX, self.circlePointY, self.circleWidth, self.circleHeight)

        # painter = QPainter()
        # painter.setOpacity(1.0)
        # painter.setBrush(QtGui.QBrush(PoleObject.unbalanceColor[self.unbalanceClass]))
        # self.unbalanceStateCircle.paint(painter, QtWidgets.QStyleOptionGraphicsItem())
        self.unbalanceStateCircle.setPen(QtGui.QPen(PoleObject.unbalanceColor[self.unbalanceClass]))
        self.unbalanceStateCircle.setBrush(QtGui.QBrush(PoleObject.unbalanceColor[self.unbalanceClass]))
        # self.unbalanceStateCircle.setPainter(painter)
        # painter.(self.unbalanceStateCircle)
        self.diagramScene.addItem(self.unbalanceStateCircle)

        parentObject = self.parentItem()
        insertTableIndex = self.calcInsesrtTableIndex()

        print('Parent Type : ', type(parentObject))
        if insertTableIndex >= 0:
            ObjectControlClass.insertUnbalanceDataIntoTable(insertTableIndex, self.getPoleID(), self.getUnbalanceClass(), self.getUnbalanceInfo())

        # self.unbalanceStateCircle.setParentItem(self)

    def calcInsesrtTableIndex(self):
        poleObjectIndexList = ObjectControlClass.tableIndexOfPoleObject.keys()
        for indexItem in poleObjectIndexList:
            if ObjectControlClass.tableIndexOfPoleObject[indexItem] == self.getPoleID():
                return int(indexItem) + 3

        return -1

    def getCenterLinePenWidth(self):
        return self.centerLinePenWidth

    def getShortLinePenWidth(self):
        return self.shortLinePenWidth

class tmpClass:
    def __init__(self):
        print('IoT 알고리즘 구현 클래스!')

    def calcPoleUnbalance(self, displayModeValue, poles):
        print('calcPoleUnbalance in')
        print('displayModeValue : ', displayModeValue)
        idx = 0
        returnDic = {}
        for poleItem in poles:
            poleObject = ObjectControlClass.poleObjectDic[poleItem]
            resultUnbalanceClass = self.calcUnbalanceClass(idx)
            # poleObject.setUnbalanceClass(resultUnbalanceClass)
            resultUnbalanceInfo = self.calcUnbalanceInfo(idx)
            # poleObject.setUnbalanceInfo(resultUnbalanceInfo)
            returnDic[poleItem] = {'unbalanceClass': resultUnbalanceClass, 'unbalanceInfo': resultUnbalanceInfo}
            idx += 1
        return returnDic

    def calcUnbalanceClass(self, idx):
        classificationValue = idx % 3
        resultValue = None

        if classificationValue == 0:
            resultValue = 'High'
        elif classificationValue == 1:
            resultValue = 'Low'
        else:
            resultValue = 'Normal'

        return resultValue

    def calcUnbalanceInfo(self, idx):
        resultValue = idx * 10

        if idx < 5:
            resultValue = idx * 100
        elif idx < 10:
            resultValue = idx * 5
        elif idx < 15:
            resultValue = idx * 3
        else:
            resultValue = idx * 2

        return resultValue

    def displayTemperatureGraph(self, displayModeValue, poleId):

        createRandomCount = 0
        dataA, dataB, dataC = None, None, None
        if displayModeValue == int(ObjectControlClass.displayModeValue['Total']):
            createRandomCount = 150
        elif displayModeValue == int(ObjectControlClass.displayModeValue['Monthly']):
            createRandomCount = 30
        elif displayModeValue == int(ObjectControlClass.displayModeValue['Daily']):
            createRandomCount = 1

        dataA = [random.random() for i in range(createRandomCount)]
        dataB = [random.random() for i in range(createRandomCount)]
        dataC = [random.random() for i in range(createRandomCount)]

        resultTemperatureData = {}
        resultTemperatureData['A'] = dataA
        resultTemperatureData['B'] = dataB
        resultTemperatureData['C'] = dataC

        #  getDialogHorizontalLayout

        # drawTemperatureGraphCanvas = temperatureGraphWindow()
        # drawTemperatureGraphCanvas.drawTemperatureGraphWindow(resultTemperatureData)

        # drawTemperatureGraphCanvas = PlotCanvas(self, width=5, height=2)
        # DiagramScene.drawTemperatureGraphCanvas.drawTemperatureGraph(resultTemperatureData)
        # sys.exit(app.exec_())

class temperatureGraphWindow(object):
    def __init__(self):
        super().__init__()
        # self.left = 10
        # self.top = 10
        # self.title = 'PyQt5 matplotlib example - pythonspot.com'
        # # 전체 윈도우의 크기 지정
        # self.width = 640
        # self.height = 400


    def drawTemperatureGraphWindow(self, graphData):
        self.setGraphData(graphData)
        self.setUi()
        # self.initUI()

    def setGraphData(self, graphData):
        self.graphData = graphData

    def setUi(self):
        # print('in 1')
        # form = QtGui.QWidget()
        # print('in 2')
        # form.setObjectName("drawTemperatureGraphForm")
        # form.resize(533, 497)
        # self.mplvl = QtGui.QWidget(form)
        # self.mplvl.setGeometry(QRect(150, 150, 251, 231))
        # self.mplvl.setObjectName("mplvl")
        self.vLayout = ObjectControlClass.dialogVerticalLayout
        # self.mplvl.setLayout(self.vLayout)
        # self.retranslateUi(Form)
        # QtCore.QMetaObject.connectSlotsByName(Form)

        print('print type data : ', type(self))
        m = PlotCanvas(self, width=5, height=2)  # 그래프가 그려지는 영역의 크기 조정
        # m.move(0,250)
        m.drawTemperatureGraph(self.graphData)
        m.bindingPlotter()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        m = PlotCanvas(self, width=5, height=2)     # 그래프가 그려지는 영역의 크기 조정
        # m.move(0,250)
        m.drawTemperatureGraph(self.graphData)

        # button = QPushButton('PyQt5 button', self)
        # button.setToolTip('This s an example button')
        # button.move(500,0)
        # button.resize(140,100)

        self.show()

class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        print(type(parent))
        # self.setParent(parent)
        # self.parent = parent
        self.parent = parent

        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        self.canvas = FigureCanvas.__init__(self, fig)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def drawTemperatureGraph(self, displayData):
        # print('in 1')
        # print(type(self.figure))
        # ObjectControlClass.dialogVerticalLayout.addWidget(self)
        # print('in 2')
        self.setDisplayData(displayData)
        self.plot()

    def setDisplayData(self, displayData):
        self.displayData = displayData

    def plot(self):
        # data = [random.random() for i in range(25)]     # 0 ~ 1 사이의 난수를 25개 생성한다.
        ax = self.figure.add_subplot(111)
        ax.plot(self.displayData['A'], 'r-')
        ax.plot(self.displayData['B'], 'r-')
        ax.plot(self.displayData['C'], 'r-')
        self.figure.add_subplot(111).plot()
        ax.set_title('PyQt Matplotlib Example')
        self.draw()

    def bindingPlotter(self):
        print(type(self.parent.vLayout))
        self.parent.vLayout.addWidget(self.canvas)



# ************************************* Main 함수 구현부 ****************************************************
app = QtWidgets.QApplication([])

poleIdArr = [
    '8132X291',
    '8132W782',
    '8232P471',
    '8232R383',
    '8232R152',
    '8132W212',
    '8132W832',
    '8232P531',
    '8132W952',
    '8132Z961',
    '8132X914',
    '8132Q911',
    '8132W231',
    '8132W122',
    '8132X921',
    '8132X152',
    '8132X122',
    '8132W621',
    '8132W981',
    '8132X601'
]
tmpArrayCnt = len(poleIdArr)
forCnt = 0

# 전주 그리기
for poleId in poleIdArr:
    createPoleWidget = ObjectControlClass()
    createPoleWidget.createPoleObject(poleId)

    forCnt += 1

    if forCnt == tmpArrayCnt:
        print("forCnt : ", forCnt, "  tmpArrayCnt : ", tmpArrayCnt)
        createPoleWidget.show()
        createPoleWidget.setTableWidgetObject()
        createPoleWidget.showFullScreen()


resultPoleValueObject = ObjectControlClass.tmpClassObj
resultPoleDataDic = resultPoleValueObject.calcPoleUnbalance(int(ObjectControlClass.displayModeValue['Total']), poleIdArr)


for poleId in poleIdArr:
    resultDataDicOfPole = resultPoleDataDic[poleId]
    poleObject = ObjectControlClass.poleObjectDic[poleId]
    print('resultDataDicOfPole[unbalanceClass] : ', resultDataDicOfPole['unbalanceClass'])
    print('resultDataDicOfPole[unbalanceInfo] : ', resultDataDicOfPole['unbalanceInfo'])
    poleObject.setUnbalanceClass(resultDataDicOfPole['unbalanceClass'])
    poleObject.setUnbalanceInfo(resultDataDicOfPole['unbalanceInfo'])
    poleObject.drawUnbalanceStateCircle()

sys.exit(app.exec())