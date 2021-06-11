import numpy as np
from PyQt5 import QtCore
from PyQt5.QtWidgets import (
    QApplication,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QTextEdit, 
    QVBoxLayout, 
    QWidget,
    QLabel, 
)
import time
import sys

class UI(object):
    def setupUI(self, MainWindow):
        self.MainWindow = MainWindow
        self.MainWindow.resize(500, 500)
        self.centralWidget = QWidget(self.MainWindow)
        self.centralWidget.setObjectName('centralwidget')
        self.centralWidgetLayout = QVBoxLayout(self.centralWidget)
        self.centralWidgetLayout.setContentsMargins(0,0,0,0)
        self.setNewCentral()

    def setNewCentral(self):
        self.centralWidgetLayout.addWidget(W())
        self.MainWindow.setCentralWidget(self.centralWidget) 

class W_UI(object):
    def setupUI(self, widget):
        self.layout = QVBoxLayout(widget)
        self.line = QLineEdit()
        self.line.textEdited.connect(self.jopa)
        self.layout.addWidget(self.line)

    def jopa(self):
        print(self.line.text()) 


class W(QWidget, W_UI):
    def __init__(self, parent=None):
        super(W, self).__init__(parent) 
        self.setupUI(self)      

class Window(QMainWindow, UI):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent) 
        self.setupUI(self)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Window()
    ex.show()
    app.exec_()