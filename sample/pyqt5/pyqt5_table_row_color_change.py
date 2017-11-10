#!/usr/bin/python3
# -*- encoding: utf-8 -*-

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys

class TableData( QAbstractTableModel ):
    def __init__( self, data = [], parent = None ):
        super( QAbstractTableModel, TableData ).__init__( self, parent )
        self.data = data

    def rowCount( self, parent ):
        return len( self.data )

    def columnCount( self, parent ):
        return 1

    def data( self, index, role ):
        if role == Qt.DisplayRole:
            return self.data[ index.row() ]
        elif role == Qt.TextAlignmentRole:
            return Qt.AlignCenter

class Table( QTableView ):

    def __init__( self, parent = None ):
        super( QTableView, Table ).__init__( self )

    def mousePressEvent( self, mpEvent ) :
        self.setStyleSheet( "QTableView{ selection-background-color: red; }" )
        QTableView.mousePressEvent( self, mpEvent )

    def mouseReleaseEvent( self, mrEvent ) :
        self.setStyleSheet( "QTableView{ selection-background-color: none; }" )
        QTableView.mouseReleaseEvent( self, mrEvent )

if __name__ == '__main__':

    app = QApplication( sys.argv )

    Gui = Table()
    Gui.setModel( TableData( ['1', '2', '3', '4', '5'], Gui ) )
    Gui.show()

    sys.exit( app.exec_() )