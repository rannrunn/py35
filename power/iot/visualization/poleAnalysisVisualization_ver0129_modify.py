import datetime
import random
import sys
import time

import MySQLdb
import numpy as np
import pandas as pd
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5 import uic
from PyQt5.QtCore import *
from PyQt5.QtGui import QColor, QPen
from PyQt5.QtWidgets import QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import dbConnection


# import unbalanceLoadInfo as uli


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

            self.chgUnbalanceCircleInfo(selectedRowNum)
        else:
            print('Not Selected')

    def chgUnbalanceCircleInfo(self, selectedRowNum):
        ObjectControlClass.selectDisplayMode = selectedRowNum
        resultCircleInfoDic = ObjectControlClass.tmpClassObj.calcPoleUnbalance(int(selectedRowNum), ObjectControlClass.tableIndexOfPoleObject.values())

        # returnDic[poleItem] = {'poleId': poleItem, 'unbalanceClass': resultUnbalanceClass, 'unbalanceInfo': resultUnbalanceInfo}

        print('resultCircleInfoDic length', len(resultCircleInfoDic))

        for poleItem in ObjectControlClass.poleObjectDic.keys():
            poleUnbalanceData = resultCircleInfoDic[poleItem]
            unbalanceClass = poleUnbalanceData['unbalanceClass']
            unbalanceInfo = poleUnbalanceData['unbalanceInfo']

            poleObject = ObjectControlClass.poleObjectDic[poleItem]

            poleObject.setUnbalanceClass(unbalanceClass)
            poleObject.setUnbalanceInfo(unbalanceInfo)
            poleObject.drawUnbalanceStateCircle()

            # for unbalanceCircleInfoItem in resultCircleInfoDic:
            #     print(unbalanceCircleInfoItem.keys())
            #     poleId = unbalanceCircleInfoItem['poleId']
            #     unbalanceClass = unbalanceCircleInfoItem['unbalanceClass']
            #     unbalanceInfo = unbalanceCircleInfoItem['unbalanceInfo']
            #
            #     poleObject = ObjectControlClass.poleObjectDic[poleId]
            #
            #     poleObject.setUnbalanceClass(unbalanceClass)
            #     poleObject.setUnbalanceInfo(unbalanceInfo)
            #     poleObject.drawUnbalanceStateCircle()



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

        if idx >= ObjectControlClass.unbalanceDataTable.rowCount():
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

        self.insertTableIndex = -1

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
        PoleObject.unbalanceInfoSize['Big'] = 45
        PoleObject.unbalanceInfoSize['Medium'] = 35
        PoleObject.unbalanceInfoSize['Small'] = 25


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
        if self.unbalanceClass == 'High':
            resultData = 'Big'
        elif self.unbalanceClass == 'Low':
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

        if self.unbalanceStateCircle is not None:
            self.diagramScene.removeItem(self.unbalanceStateCircle)

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
        if self.insertTableIndex == -1:
            self.insertTableIndex = self.calcInsesrtTableIndex()

        print('Parent Type : ', type(parentObject))
        if self.insertTableIndex >= 0:
            ObjectControlClass.insertUnbalanceDataIntoTable(self.insertTableIndex, self.getPoleID(), self.getUnbalanceClass(), self.getUnbalanceInfo())

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
        # self.obj_uli = uli.UnbalanceLoadInfo()
        self.date = '2017-07-01'
        self.file_name = './predict_all.csv'
        self.df = pd.read_csv(self.file_name)

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

            resultUnbalanceClass = 0
            resultUnbalanceInfo = 0
            resultUnbalanceClassTemp = 0

            if displayModeValue == 2:
                resultUnbalanceInfo, resultUnbalanceClassTemp = UnbalanceLoadInfo().getDailyInfoCSV(self.df, poleItem, self.date)
            elif displayModeValue == 1:
                resultUnbalanceInfo, resultUnbalanceClassTemp = UnbalanceLoadInfo().getMonthlyInfoCSV(self.df, poleItem, self.date)
            elif displayModeValue == 0:
                resultUnbalanceInfo, resultUnbalanceClassTemp = UnbalanceLoadInfo().getTotalInfoCSV(self.df, poleItem, self.date)

            print('unbalanceClass:', resultUnbalanceClassTemp, 'unbalanceInfo:', resultUnbalanceInfo)

            if resultUnbalanceClassTemp == 0:
                resultUnbalanceClass = 'Normal'
            elif resultUnbalanceClassTemp == 1:
                resultUnbalanceClass = 'Low'
            elif resultUnbalanceClassTemp == 2:
                resultUnbalanceClass = 'High'

            returnDic[poleItem] = {'poleId': poleItem, 'unbalanceClass': resultUnbalanceClass, 'unbalanceInfo': resultUnbalanceInfo}

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

class UnbalanceLoadInfo():
    def __init__(self):
        print('머신시작1-1')
        # DB connection
        self.con = dbConnection.getConnection()
        self.con.set_character_set('utf8')
        self.cur = self.con.cursor(MySQLdb.cursors.DictCursor)
        self.cur.execute('SET NAMES utf8;')
        self.cur.execute('SET CHARACTER SET utf8;')
        self.cur.execute('SET character_set_connection=utf8;')
        self.path_dir = 'F:\\IOT\\data\\2'

    # SELECT
    def selecPoleTemp(self, sensor_id, time_start, time_end):
        query = """SELECT TIME_ID, TEMP
                    FROM TB_IOT_POLE_SECOND s1
                    WHERE SENSOR_ID = '%s'
                    AND TIME_ID BETWEEN '%s' AND '%s'
                    """ % (sensor_id, time_start, time_end)
        self.cur.execute(query)
        results = self.cur.fetchall()
        df = pd.DataFrame(list(results))
        return df


    # SELECT
    def selectSensorList(self, pole_id):
        query = """SELECT POLE_ID, SENSOR_ID, PART_NAME
                    FROM TB_IOT_POLE_INFO s1
                    WHERE POLE_ID = '%s'
                    AND IFNULL(POLE_ID, '') != ''
                    AND IFNULL(SENSOR_ID, '') != ''
                    AND IFNULL(PART_NAME, '') != ''
                    """ % (pole_id)
        self.cur.execute(query)
        results = self.cur.fetchall()
        df = pd.DataFrame(list(results))
        return df

    def getMonthlyInfoDB(self, pole_id, time_start, time_end):
        print(time_start)
        print(time_end)
        index_start = datetime.datetime.strptime(time_start[:19], '%Y-%m-%d %H:%M:%S')
        index_end = datetime.datetime.strptime(time_end[:19], '%Y-%m-%d %H:%M:%S')
        time_start_predict = datetime.datetime(index_start.year, index_start.month, index_start.day, 0, 0, 0)
        time_end_predict = time_start_predict + datetime.timedelta(1) - datetime.timedelta(seconds=1)

        unbalanceCount = 0
        unbalanceClass = 0

        cnt = 0
        while time_start_predict < index_end:
            time_start_predict += datetime.timedelta(days=1)
            time_end_predict += datetime.timedelta(days=1)
            unbalanceCount += self.getDailyInfo(pole_id, time_start_predict, time_end_predict)
            # print('unbalanceCount:', unbalanceCount)
            cnt += 1
            # print('cnt:', cnt)

        if unbalanceCount > 10:
            unbalanceClass = 2
        elif unbalanceCount > 5:
            unbalanceClass = 1

        return unbalanceCount, unbalanceClass


    def getDailyInfoDB(self, pole_id, time_start, time_end):

        df = self.getData(pole_id, time_start, time_end)
        arr = self.setInput(df)
        unbalanceCount = 0
        # print('arr:', len(arr))
        if len(arr) == 72:
            # print('예측시작')
            unbalanceCount = self.model.predict(arr)

        return unbalanceCount

    def getTotalInfoCSV(self, df, pole_id, date):
        df = df[df['POLE_ID'] == pole_id]

        index_start = datetime.datetime.strptime('2017-06-01 00:00:00', '%Y-%m-%d %H:%M:%S')
        index_end = datetime.datetime.strptime('2017-12-31 23:59:59', '%Y-%m-%d %H:%M:%S')
        time_start_predict = datetime.datetime(index_start.year, index_start.month, index_start.day, 0, 0, 0)
        time_end_predict = time_start_predict + datetime.timedelta(1) - datetime.timedelta(seconds=1)

        unbalanceCount = 0
        unbalanceClass = 0

        cnt = 0
        while time_start_predict < index_end:
            unbalanceCountDaily, unbalanceClassDaily = self.getDailyInfoCSV(df, pole_id, str(time_start_predict)[0:10])
            time_start_predict += datetime.timedelta(days=1)
            time_end_predict += datetime.timedelta(days=1)
            # print('unbalanceCount:', unbalanceCount)
            unbalanceCount += unbalanceCountDaily
            cnt += 1

        if unbalanceCount > 70:
            unbalanceClass = 2
        elif unbalanceCount > 50:
            unbalanceClass = 1

        return unbalanceCount, unbalanceClass

    def getMonthlyInfoCSV(self, df, pole_id, date):
        df = df[df['POLE_ID'] == pole_id]

        index_start = datetime.datetime.strptime('2017-07-01 00:00:00', '%Y-%m-%d %H:%M:%S')
        index_end = datetime.datetime.strptime('2017-07-31 23:59:59', '%Y-%m-%d %H:%M:%S')
        time_start_predict = datetime.datetime(index_start.year, index_start.month, index_start.day, 0, 0, 0)
        time_end_predict = time_start_predict + datetime.timedelta(1) - datetime.timedelta(seconds=1)

        unbalanceCount = 0
        unbalanceClass = 0

        cnt = 0
        while time_start_predict < index_end:
            unbalanceCountDaily, unbalanceClassDaily = self.getDailyInfoCSV(df, pole_id, str(time_start_predict)[0:10])
            time_start_predict += datetime.timedelta(days=1)
            time_end_predict += datetime.timedelta(days=1)
            # print('unbalanceCount:', unbalanceCount)
            unbalanceCount += unbalanceCountDaily
            cnt += 1

        if unbalanceCount > 10:
            unbalanceClass = 2
        elif unbalanceCount > 5:
            unbalanceClass = 1

        return unbalanceCount, unbalanceClass

    def getDailyInfoCSV(self, df, pole_id, date):
        df = df[df['POLE_ID'] == pole_id]
        df = df[df['DATE'] == date]
        resultClass = 0
        if df['UNBALANCE_FLAG'].values[0] == 1:
            resultClass = 1
        return df['UNBALANCE_FLAG'].values[0], resultClass

    def setInput(self, df):
        arr = np.array([])
        for item in df.columns.values:
            arr = np.append(arr, df[item].values)
        for idx in range(3 - len(df.columns.values)):
            arr = np.append(arr, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        return arr


    def getData(self, pole_id, time_start, time_end):
        start = time.time()

        df_result = pd.DataFrame(columns=['TIME_ID'])
        df_result.set_index('TIME_ID', inplace=True)

        df_sensor = self.selectSensorList(pole_id)
        for idx in range(len(df_sensor)):
            sensor_id = str(df_sensor['SENSOR_ID'][idx])
            part_name = str(df_sensor['PART_NAME'][idx])
            if part_name == 'nan' or part_name != '변압기 본체':
                continue
            df_temp = self.selecPoleTemp(sensor_id, time_start, time_end)
            if len(df_temp) == 0:
                continue
            df_temp = df_temp[['TIME_ID', 'TEMP']]
            df_temp['TEMP'] = df_temp['TEMP'].astype(float)
            df_temp.columns = ['TIME_ID', sensor_id]
            df_temp.set_index('TIME_ID', inplace=True)
            df_temp.index = pd.to_datetime(df_temp.index)

            df_temp = df_temp.resample('60T').mean()
            df_result = pd.merge(df_result, df_temp, left_index=True, right_index=True, how='outer')

        # mask = (df['TIME_ID'] >=  '') & (df['TIME_ID'] <= '')
        # df = df.loc[mask],ㅣㅣㅣㅣㅣ

        print(time.time() - start)

        return df_result


    def getListPole(self):
        return  [
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


    def writePredictCsv(self):

        time_start = '2017-06-01 00:00:00'
        time_end = '2017-12-31 23:59:59'

        list_pole = self.getListPole()

        obj = UnbalanceLoadInfo()

        df_result = pd.DataFrame(columns=['DATE', 'POLE_ID', 'UNBALANCE_FLAG'])

        for pole_id in list_pole:
            index_start = datetime.datetime.strptime(time_start[:19], '%Y-%m-%d %H:%M:%S')
            index_end = datetime.datetime.strptime(time_end[:19], '%Y-%m-%d %H:%M:%S')
            time_start_predict = datetime.datetime(index_start.year, index_start.month, index_start.day, 0, 0, 0)
            time_end_predict = time_start_predict + datetime.timedelta(1) - datetime.timedelta(seconds=1)

            df = obj.getData(pole_id, time_start, time_end)

            cnt = 0
            while time_start_predict < index_end:

                df_data = df.loc[time_start_predict:time_end_predict,:]

                arr = obj.setInput(df_data)
                unbalanceFlag = 0
                # print('arr:', len(arr))
                if len(arr) == 72:
                    # print('예측시작')
                    unbalanceFlag = obj.model.predict(arr)
                print(time_start_predict)
                df_result = df_result.append({'DATE': time_start_predict, 'POLE_ID':pole_id, 'UNBALANCE_FLAG':unbalanceFlag}, ignore_index=True)

                time_start_predict += datetime.timedelta(days=1)
                time_end_predict += datetime.timedelta(days=1)

                print('unbalanceFlag:', unbalanceFlag)
                cnt += 1
                # print('cnt:', cnt)

        # print(df_result)
        df_result.set_index('DATE', inplace=True)
        df_result.to_csv('predict_all.csv')

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
    # print('resultDataDicOfPole[unbalanceClass] : ', resultDataDicOfPole['unbalanceClass'])
    # print('resultDataDicOfPole[unbalanceInfo] : ', resultDataDicOfPole['unbalanceInfo'])
    poleObject.setUnbalanceClass(resultDataDicOfPole['unbalanceClass'])
    poleObject.setUnbalanceInfo(resultDataDicOfPole['unbalanceInfo'])
    poleObject.drawUnbalanceStateCircle()

sys.exit(app.exec())