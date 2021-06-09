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


class UI1(object):
    signal = QtCore.pyqtSignal()
    def setupUI(self, widget):
        self.layout = QVBoxLayout(widget)
        label = QLabel()
        label.setText('widget1')
        button = QPushButton()
        button.setText('widget2')
        button.clicked.connect(lambda: self.signal.emit())
        self.layout.addWidget(label)
        self.layout.addWidget(button)

class UI2(object):
    signal = QtCore.pyqtSignal(str)
    def setupUI(self, widget):
        self.layout = QVBoxLayout(widget)
        self.label = QLabel()
        self.line = QLineEdit(self)
        self.line.textEdited.connect(lambda: self.update())
        self.line.setText('jopa')

        self.layout.addWidget(self.line)
        self.layout.addWidget(self.label)

    def update(self):
        print('jopa')

class Widget1(QWidget, UI1):
    def __init__(self, parent=None):
        super(Widget1, self).__init__(parent) 
        self.setupUI(self)

class Widget2(QWidget, UI2):
    def __init__(self, parent=None):
        super(Widget2, self).__init__(parent) 
        self.setupUI(self)


class UI(object):
    def setupUI(self, MainWindow):
        self.MainWindow = MainWindow
        self.MainWindow.resize(500, 500)
        self.centralWidget = QWidget(self.MainWindow)
        self.centralWidget.setObjectName('centralwidget')
        self.centralWidgetLayout = QVBoxLayout(self.centralWidget)
        self.centralWidgetLayout.setContentsMargins(0,0,0,0)
        widget = Widget1()
        widget.signal.connect(self.load_widget2)
        self.setNewCentral(widget)

    def setNewCentral(self, widget):
        self.clearLayout(self.centralWidgetLayout)
        self.centralWidgetLayout.addWidget(widget)
        self.MainWindow.setCentralWidget(self.centralWidget)

    def load_widget2(self):
        widget = Widget2()
        self.setNewCentral(widget)

    def clearLayout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()    

class Window(QMainWindow, UI):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent) 
        self.setupUI(self)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Window()
    ex.show()
    app.exec_()