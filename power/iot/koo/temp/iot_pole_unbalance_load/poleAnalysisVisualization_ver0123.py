import sys
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5.QtGui import QPainter, QColor, QPen
from PyQt5.QtCore import *
from PyQt5 import uic
import unbalanceLoadInfo as uli


# GraphicsScene Event용 class
class DiagramScene(QtWidgets.QGraphicsScene):
    # itemSelected = pyqtSignal(QtWidgets.QGraphicsItem)

    def __init__(self, parent=None):
        super(DiagramScene, self).__init__(parent)

    def mousePressEvent(self, mouseEvent):
        if (mouseEvent.button() != Qt.LeftButton):
            return

        selectedItems = self.items(mouseEvent.scenePos())  # 선택된 객체 목록을 리스트 형태로 반환
        # 선택된 항목 리스트 중 첫번째 객체 획득
        # selectedItem.getCircleLocation()
        # if type(selectedItem) == QtWidgets.QGraphicsEllipseItem.type():
        # print(type(selectedItem))
        # print(circleGraphicObject)
        if selectedItems.__len__() > 0:
            selectedItem = selectedItems[0]

            # print(type(selectedItem))

            # QLabel Object 선택 시 return 처리
            # if type(selectedItem) == QtWidgets.QGraphicsProxyWidget:
            #     # print(type(selectedItem))
            #     # print('label in')
            #     return
            # elif type(selectedItem) == QtWidgets.QGraphicsSimpleTextItem:
            #     # print('in')
            #     print('click widget info : ')
            #     selectedItem.parentItem().getLocation()
            #     selectedItem.parentItem().toggleState()
            #     # print(type(selectedItem.parentItem()))
            #     return
            #
            # selectedItem.toggleState()
            # selectedItem.testingFunc()
            super(DiagramScene, self).mousePressEvent(mouseEvent)

            # if type(selectedItem) == circleGraphicObject:
            #     print('in circle object click event')
            #     selectedItem.toggleState()
            #     super(DiagramScene, self).mousePressEvent(mouseEvent)
            # elif type(selectedItem) == rectGraphicObject:
            #     print('in rect object click event')
            # elif type(selectedItem) == lineGraphicObject:
            #     print('in line object click event')
            #     selectedItem.toggleState()
            #     super(DiagramScene, self).mousePressEvent(mouseEvent)
        else:
            # print('else object click')
            return

        super(DiagramScene, self).mousePressEvent(mouseEvent)


class ObjectControlClass(QtWidgets.QDialog):
    diagramScene = None
    unbalanceDataTable = None
    poleObjectDic = {}
    tableIndexOfPoleObject = {}
    betweenObjectMarginValue = {}
    lineDictionary = {}
    LastIndex = -1

    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.ui = uic.loadUi("IoTPoleGraphicsView.ui", self)
        self.dialogHorizontalLayout = self.ui.dialogHorizontalLayout

        self.unbalanceDataTable = None  # 불균형 정보를 저장할 테이블

        self.setBetweenObjectMarginValue()
        self.setUi()

    def setBetweenObjectMarginValue(self):

        topValue = [0, -130]
        leftValue = [-45, 0]
        rightValue = [45, 0]
        bottomValue = [0, 130]

        ObjectControlClass.betweenObjectMarginValue['Top'] = topValue
        ObjectControlClass.betweenObjectMarginValue['Bottom'] = bottomValue
        ObjectControlClass.betweenObjectMarginValue['Left'] = leftValue
        ObjectControlClass.betweenObjectMarginValue['Right'] = rightValue

    def setUi(self):
        # diagramScene = DiagramScene()
        #
        # tableWidget = QtWidgets.QTableWidget()
        # self.scene = DiagramScene()
        # self.diagramGraphicsView.setScene(scene)

        if ObjectControlClass.diagramScene == None:
            ObjectControlClass.diagramScene = DiagramScene()

        self.diagramGraphicsView.setScene(ObjectControlClass.diagramScene)
        self.setMouseTracking(True)

        if ObjectControlClass.unbalanceDataTable == None:
            ObjectControlClass.unbalanceDataTable = QtWidgets.QTableWidget()
            self.setTableInfo()
            # self.setTableWidgetObject()
            # self.initDiagramTableWidget()
            # ObjectControlClass.diagramTableWidget.itemChanged.connect(self.tableItemChangedEvent)
            # ObjectControlClass.diagramTableWidget.itemClicked.connect(self.tableItemClickedEvent)
            ObjectControlClass.unbalanceDataTable.setMinimumWidth(350)
            ObjectControlClass.unbalanceDataTable.setMaximumWidth(350)
            ObjectControlClass.unbalanceDataTable.verticalHeader().setVisible(False)

    def setTableInfo(self):
        self.setTableWidgetObject()  # 테이블을 Graphic Scene에 등록
        self.initDiagramTableWidget()  # Table의 스펙(헤더의 수와 값, 크기) 지정

    def setTableWidgetObject(self):
        self.dialogHorizontalLayout.addWidget(ObjectControlClass.unbalanceDataTable)
        self.setLayout(self.dialogHorizontalLayout)

    def initDiagramTableWidget(self):
        ObjectControlClass.unbalanceDataTable.resize(10, 30)
        ObjectControlClass.unbalanceDataTable.setColumnCount(3)
        ObjectControlClass.unbalanceDataTable.setRowCount(0)
        ObjectControlClass.unbalanceDataTable.setHorizontalHeaderLabels(['Pole Id', 'unbalanceClass', 'unbalanceInfo'])

    # 객체 생성을 위해 별도로 호출하는 함수
    def createPoleObject(self, poleID):
        self.poleObject = PoleObject(poleID)
        tmpPointX, tmpPointY = self.calcStandardPoint()
        self.poleObject.setLocation(tmpPointX, tmpPointY)
        self.poleObject.drawPoleWidget()

        self.insertPoleObjectIntoDictionary()

        self.poleObject.addPoleWidget(ObjectControlClass.diagramScene)

        # ObjectControlClass.diagramScene.addWidget(self.poleObject)

    def calcStandardPoint(self):
        tmpPointX, tmpPointY = 0, 0
        print(len(self.poleObjectDic))
        if len(self.poleObjectDic) == 0:
            tmpPointX = 250
            tmpPointY = 150
        else:
            parentIdex = (int(ObjectControlClass.LastIndex / 3) - 1) * 3 + 1
            if parentIdex < 0:
                parentIdex = 0
            print('parentIdex : ', parentIdex)
            print('LastIndex + 1  : ', ObjectControlClass.LastIndex + 1)
            parentPoleId = ObjectControlClass.tableIndexOfPoleObject[parentIdex]
            parentPoleObject = ObjectControlClass.poleObjectDic[parentPoleId]
            parentPointX, parentPointY = parentPoleObject.getLocation()
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            gapLocation = ObjectControlClass.betweenObjectMarginValue[self.setPoleDirection()]
            tmpPointX = parentPointX + gapLocation[0]
            tmpPointY = parentPointY + gapLocation[1]

            print('Currunt Pole Point : ( ', tmpPointX, ', ', tmpPointY, ')')

        return tmpPointX, tmpPointY

    def setPoleDirection(self):
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

    def insertPoleObjectIntoDictionary(self):
        ObjectControlClass.LastIndex += 1
        ObjectControlClass.tableIndexOfPoleObject[ObjectControlClass.LastIndex] = self.poleObject.getPoleID()
        ObjectControlClass.poleObjectDic[self.poleObject.getPoleID()] = self.poleObject

    # unbalanceDataTable에 데이터를 넣어주는 부분
    # Pole 객체 생성 시에 함께 넣어준다.
    def insertUnbalanceDataIntoTable(self, idx, unbalanceClass, unbalanceInfo):
        ObjectControlClass.unbalanceDataTable.insertRow(idx)
        ObjectControlClass.unbalanceDataTable.setItem(idx, 0, unbalanceClass)
        ObjectControlClass.unbalanceDataTable.setItem(idx, 1, unbalanceInfo)


class PoleObject(QtWidgets.QGraphicsLineItem, QtWidgets.QGraphicsEllipseItem):
    def __init__(self, poleID):
        super(PoleObject, self).__init__()

        self.poleID = poleID
        self.unbalanceClass = None  # 고위험군, 저위험군, 일반
        self.unbalanceInfo = None  # 불균형 횟수 데이터

        self.centerLineObject = None
        self.shortFirstLineObject = None
        self.shortSecondLineObject = None
        self.unbalanceStateCircle = None

        self.parentPole = None
        self.childPoles = {}
        self.nodeDirection = None  # 부모에 대해 상대적인 위치 정보를 저장

        PoleObject.unbalanceClassValue = {}
        PoleObject.unbalanceInfoValue = {}
        PoleObject.unbalanceInfoSize = {}
        PoleObject.unbalanceColor = {}

        PoleObject.nodeDirectionValue = ['RIGHT', 'UP', 'DOWN', 'LEFT']
        PoleObject.setClassificationValue()
        self.setPoleDesign()

    def setPoleDesign(self):
        PoleObject.centerLineLength = 100
        PoleObject.shortLineLength = 30

        PoleObject.firstShortLineY = 30
        PoleObject.gapOfShortLineY = 15

        # 원을 그릴 때 사용할 원의 지름을 지정
        PoleObject.unbalanceInfoSize['Big'] = 60
        PoleObject.unbalanceInfoSize['Medium'] = 40
        PoleObject.unbalanceInfoSize['Small'] = 20

        # 불균형 위험군 분류에 따른 원의 색상 지정
        PoleObject.unbalanceColor['High'] = QtGui.QBrush(QColor(Qt.red))
        # PoleObject.unbalanceColor['Low'] = QtGui.QBrush(QColor(Qt.yellow))
        PoleObject.unbalanceColor['Low'] = QtGui.QBrush(QColor(QColor(253, 126, 23)))
        PoleObject.unbalanceColor['Normal'] = QtGui.QBrush(QColor(Qt.green))

        self.drawCenterLinePen = QtGui.QPen(QColor(Qt.darkGray))
        self.drawCenterLinePen.setWidth(10)

        self.drawShortLinePen = QtGui.QPen(QColor(Qt.darkGray))
        self.drawShortLinePen.setWidth(8)

    def displayPoleInfo(self):
        print('************ Pole Location Info ************')
        print(' # Pole ID : ', self.poleID)
        print(' # Pole Point : ( ', self.pointX, ', ', self.pointY, ' )')
        print(' # Unbalance Class : ', self.unbalanceClass)
        print(' # Unbalance Info : ', self.unbalanceInfo)
        print()

    def getLocation(self):
        self.displayPoleInfo()
        return self.pointX, self.pointY

    def setLocation(self, pointX, pointY):
        self.pointX, self.pointY = pointX, pointY
        self.displayPoleInfo()

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
        self.centerLineObject = QLineF(self.centerX, self.centerY, self.centerX,
                                       self.centerY + PoleObject.centerLineLength)
        self.shortFirstLineObject = QLineF(self.firstLineX, self.firstLineY,
                                           self.firstLineX + PoleObject.shortLineLength, self.firstLineY)
        self.shortSecondLineObject = QLineF(self.secondLineX, self.secondLineY,
                                            self.secondLineX + PoleObject.shortLineLength, self.secondLineY)

        self.setLine(self.centerLineObject)
        self.setLine(self.shortFirstLineObject)
        self.setLine(self.shortSecondLineObject)

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
        self.circleWidth, self.circleHeight = PoleObject.unbalanceInfoSize[self.unbalanceInfoSizeValue], \
                                              PoleObject.unbalanceInfoSize[self.unbalanceInfoSizeValue]
        self.circleCenterPointX, self.circleCenterPointY = self.pointX + (
        PoleObject.shortLineLength / 2), self.pointY + (PoleObject.gapOfShortLineY / 2)
        self.circlePointX, self.circlePointY = self.circleCenterPointX - (
        self.circleWidth / 2), self.circleCenterPointY - (self.circleHeight / 2)
        self.unbalanceStateCircle = QtWidgets.QGraphicsEllipseItem(self.circlePointX, self.circlePointY,
                                                                   self.circleWidth, self.circleHeight)
        self.unbalanceStateCircle.setBrush(PoleObject.unbalanceColor[self.unbalanceClass])
        self.diagramScene.addItem(self.unbalanceStateCircle)


class tmpClass:
    def __init__(self):
        print('머신시작')
        self.unbalanceLoadModule = uli.PoleInfo()
        print('이건 매니저님이 만드실 함수를 담을 깡통 클래스입니다!')

    def calcPoleUnbalance(self, poles):
        idx = 0
        returnDic = {}

        mode = 1
        print('효0')
        print(len(poles))
        if mode == 0:
            pass
        elif mode == 1:
            time_start = '2017-07-01 00:00:00'
            time_end = '2017-07-31 23:59:59'
            cnt = 0
            for poleItem in poles:
                poleObject = ObjectControlClass.poleObjectDic[poleItem]
                resultUnbalanceClass = self.calcUnbalanceClass(idx)
                # poleObject.setUnbalanceClass(resultUnbalanceClass)
                resultUnbalanceInfo = self.calcUnbalanceInfo(idx)
                # poleObject.setUnbalanceInfo(resultUnbalanceInfo)
                # returnDic[poleItem] = {'unbalanceClass': resultUnbalanceClass, 'unbalanceInfo': resultUnbalanceInfo}
                print('효1')
                unbalanceCount, unbalanceClass = self.unbalanceLoadModule.getMonthlyInfo(poleItem, time_start, time_end)
                returnClass = ''
                if unbalanceClass == 0:
                    returnClass = 'Normal'
                elif unbalanceClass == 1:
                    returnClass = 'Low'
                elif unbalanceClass == 2:
                    returnClass = 'High'
                returnDic[poleItem] = {'unbalanceClass': returnClass, 'unbalanceInfo': unbalanceCount}
                print('효2')
                cnt += 1
                print(cnt)
                idx += 1
        # 일
        elif mode == 2:
            time_start = '2017-07-01 00:00:00'
            time_end = '2017-07-01 23:59:59'
            for poleItem in poles:
                poleObject = ObjectControlClass.poleObjectDic[poleItem]
                resultUnbalanceClass = self.calcUnbalanceClass(idx)
                # poleObject.setUnbalanceClass(resultUnbalanceClass)
                resultUnbalanceInfo = self.calcUnbalanceInfo(idx)
                # poleObject.setUnbalanceInfo(resultUnbalanceInfo)
                # returnDic[poleItem] = {'unbalanceClass': resultUnbalanceClass, 'unbalanceInfo': resultUnbalanceInfo}
                unbalanceCount = self.unbalanceLoadModule.getDailyInfo(poleItem, time_start, time_end)
                unbalanceClass = 0
                returnClass = ''
                if unbalanceClass == 0:
                    returnClass = 'Normal'
                elif unbalanceClass == 1:
                    returnClass = 'Low'
                elif unbalanceClass == 2:
                    returnClass = 'High'
                returnDic[poleItem] = {'unbalanceClass': returnClass, 'unbalanceInfo': unbalanceCount}
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

print('이전-1')
resultPoleValueObject = tmpClass()
print('이전0')
resultPoleDataDic = resultPoleValueObject.calcPoleUnbalance(poleIdArr)

for poleId in poleIdArr:
    resultDataDicOfPole = resultPoleDataDic[poleId]
    poleObject = ObjectControlClass.poleObjectDic[poleId]
    print('resultDataDicOfPole[unbalanceClass] : ', resultDataDicOfPole['unbalanceClass'])
    print('resultDataDicOfPole[unbalanceInfo] : ', resultDataDicOfPole['unbalanceInfo'])
    poleObject.setUnbalanceClass(resultDataDicOfPole['unbalanceClass'])
    poleObject.setUnbalanceInfo(resultDataDicOfPole['unbalanceInfo'])
    poleObject.drawUnbalanceStateCircle()

sys.exit(app.exec())
