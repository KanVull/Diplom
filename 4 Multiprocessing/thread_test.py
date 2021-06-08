from PyQt5 import QtCore
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow, 
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
        self.label = QLabel()
        self.label.setText('Checking.')
        self.centralWidgetLayout.addWidget(self.label)
        self.MainWindow.setCentralWidget(self.centralWidget)

        self.startThread()

    def startThread(self):
        self.threadWork = QtCore.QThread()
        self.threadAnim = AnimationThread()
        self.QObject_work = Work()
        self.QObject_work.moveToThread(self.threadWork)

        self.threadWork.started.connect(self.QObject_work.run)
        self.QObject_work.work_done.connect(self.threadAnim.requestInterruption)
        self.threadAnim.time_passes.connect(self.updateLabel)
        self.threadAnim.finished.connect(self.debug)

        self.threadAnim.start()
        self.threadWork.start()

    def debug(self):
        print('fkewfoje')

    def updateLabel(self):
        if self.label.text().endswith('...'):
            self.label.setText('Checking.')
        else:
            self.label.setText(self.label.text() + '.')    

class AnimationThread(QtCore.QThread):
    time_passes = QtCore.pyqtSignal()
    def __init__(self):
        QtCore.QThread.__init__(self)

    def run(self):
        while not self.isInterruptionRequested():
            self.time_passes.emit()
            time.sleep(0.5)
        print('finished')        

class Work(QtCore.QObject):
    work_done = QtCore.pyqtSignal()

    def __init__(self):
        QtCore.QObject.__init__(self)

    def run(self):
        time.sleep(5)
        print('finished')
        self.work_done.emit()


class Window(QMainWindow, UI):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent) 
        self.setupUI(self)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Window()
    ex.show()
    app.exec_()