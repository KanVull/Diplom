from multiprocessing import Pool
import os
import sys
import threading

from PyQt5 import uic, QtWidgets
from PyQt5.QtCore import pyqtSignal
signal = pyqtSignal()


def fetchdata(value):
    num, message = value
    signal.emit()
    return True


def processchain(message):
    p = Pool(processes=15)
    data = p.map(fetchdata, [(i, message) for i in range(1, 1000)])
    print("results:", data)


def alltask(message):
    threading.Thread(target=processchain, args=(message,), daemon=True).start()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = QtWidgets.QMainWindow()
    widget = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(widget)
    line = QtWidgets.QLineEdit()
    button = QtWidgets.QPushButton()
    layout.addWidget(line)
    layout.addWidget(button)
    window.setCentralWidget(widget)
    signal.connect(lambda: print('JOPA'))

    def on_clicked():
        message = line.text()
        alltask(message)

    button.clicked.connect(on_clicked)
    window.show()

    sys.exit(app.exec_())