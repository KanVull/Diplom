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
        self.line = QLineEdit()
        self.line.textEdited.connect(self.jopa)
        self.setNewCentral(self.line)

    def setNewCentral(self, widget):
        self.centralWidgetLayout.addWidget(widget)
        self.MainWindow.setCentralWidget(self.centralWidget)

    def jopa(self):
        print(self.line.text())    


class Window(QMainWindow, UI):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent) 
        self.setupUI(self)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Window()
    ex.show()
    app.exec_()