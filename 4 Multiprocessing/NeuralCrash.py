import numpy as np
import multiprocessing
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow, 
    QDialog, 
    QDialogButtonBox,
    QMessageBox,
    QVBoxLayout, 
    QHBoxLayout,
    QFormLayout, 
    QWidget,
    QPushButton,
    QAction, 
    QScrollArea, 
    QLabel, 
    QCheckBox,
    QLineEdit,
    QPlainTextEdit,
)
import os
import sys
import time

from tensorflow.python.platform.tf_logging import error

class NeuralCrach():
    def __init__(self):
        self.model = None
        self.sigma = None

    def _random_numpy_normal(self, values: list, sigma):
        return [np.random.normal(array, sigma) for array in values]    

    def _create_new_model(self, sigma):
        model = self.model
        weights = self.get_values_from_model(model)
        weights = self._random_numpy_normal(weights, sigma)
        return self.set_values_to_model(weights, model)

    def _compute_models(self, sigmas):
        data = list()
        for sigma in range(int(sigmas[0]), int(sigmas[1])):
            data.append((sigma / 10000, self.evaluate_model(self._create_new_model(sigma / 10000), self.testdata)))
        return data  


    def run(self):
        app = QApplication(sys.argv)
        ex = _NeuralCrashWindow(self)
        ex.show()
        app.exec_()

        # cpu_count = multiprocessing.cpu_count() - 1
        # sigma *= 10000
        # sigma_step = int(sigma / cpu_count)
        # list_in_data = list()
        # sigma_start = 0
        # while sigma_start < sigma:
        #     sigma_end = sigma_start + sigma_step
        #     if sigma_end > sigma - sigma_step:
        #         sigma_end = sigma + 1
        #     list_in_data.append((sigma_start, sigma_end))
        #     if sigma_end >= sigma:
        #         break
        #     sigma_start += sigma_step

        # if self.model is not None:
        #     with multiprocessing.Pool(cpu_count) as p:
        #         data = p.map(self._compute_models, list_in_data)
        # else:
        #     data = None
        #     print('Model not loaded')
        # print(data)

class _NeuralCrashWindowUI(object):
    def clearLayout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()    

    def setupUI(self, MainWindow, neuralconfig):
        self.neuralconfig = neuralconfig
        self.MainWindow = MainWindow
        self.MainWindow.setWindowTitle('Test your neural')
        self.MainWindow.resize(500, 500)

        self.centralWidget = QWidget(self.MainWindow)
        self.centralWidgetLayout = QVBoxLayout(self.centralWidget)
        self.centralWidgetLayout.setContentsMargins(0,0,0,0)
        
        self.nameofpage = QLabel()
        self.contentlayout = QVBoxLayout()
        
        self.setnewcentralwidget('Checking override functions', self.create_check_layout())
        self.check_function_load_model()

    def setnewcentralwidget(self, name, layout):
        self.clearLayout(self.centralWidgetLayout)
        self.nameofpage.setText(name)
        self.contentlayout.addLayout(layout)
        self.centralWidgetLayout.addWidget(self.nameofpage)
        self.centralWidgetLayout.addLayout(self.contentlayout)
        self.MainWindow.setCentralWidget(self.centralWidget)
        self.MainWindow.setContentsMargins(0, 0, 0, 0)

    def create_check_layout(self):
        layout = QVBoxLayout()
        status = QFormLayout()
        self.error = QPlainTextEdit()
        self.error.setReadOnly(True)
        self.label_load_model = QLabel()
        self.label_load_model.setText('Checking')
        self.label_load_testdata = QLabel()
        self.label_load_testdata.setText('Checking')
        self.label_get_tested_values = QLabel()
        self.label_get_tested_values.setText('Checking')
        self.label_set_tested_values = QLabel()
        self.label_set_tested_values.setText('Checking')
        self.label_test_model = QLabel()
        self.label_test_model.setText('Checking')
        self.status_load_model = ''
        self.status_load_testdata = ''
        self.status_get_tested_values = ''
        self.status_set_tested_values = ''
        self.status_test_model = ''

        status.addRow('load_model',         self.label_load_model)
        status.addRow('load_testdata',      self.label_load_testdata)
        status.addRow('get_tested_values',  self.label_get_tested_values)
        status.addRow('set_tested_values',  self.label_set_tested_values)
        status.addRow('test_model',         self.label_test_model)

        layout.addLayout(status)
        layout.addWidget(self.error)
        return layout

    def errorprint(self, text, checked_functions):
        self.error.appendPlainText(checked_functions + '\n' + text + '\n\n\t--------\n\n')

    def update_status(self, label_name, status):
        if label_name == 'load_model':
            self.status_load_model = status
            self.label_load_model.setText(status)
        elif label_name == 'load_testdata':
            self.status_load_testdata = status
            self.label_load_testdata.setText(status)
        elif label_name == 'get_tested_values':
            self.status_get_tested_values = status
            self.label_get_tested_values.setText(status)
        elif label_name == 'set_tested_values':
            self.status_set_tested_values = status
            self.label_set_tested_values.setText(status)
        elif label_name == 'test_model':
            self.status_test_model = status
            self.label_test_model.setText(status)

    def update_label(self, label):
        if label.text().endswith('...'):
            label.setText('Checking.')
        else:
            label.setText(label.text() + '.')    

    def update_NeuralCrash(self, new_state):
        self.neuralconfig = new_state

    def check_function_load_model(self):
        self.threadAnim_load_model = CheckingAnimation()
        self.threadWork_load_model = QtCore.QThread()
        self.QObject_load_model = Check_function(self.neuralconfig, 'load_model')
        self.QObject_load_model.moveToThread(self.threadWork_load_model)

        self.threadWork_load_model.started.connect(self.QObject_load_model.run)
        self.QObject_load_model.function_checked_signal.connect(self.update_status)
        self.QObject_load_model.error_signal.connect(self.errorprint)
        self.QObject_load_model.return_new_state.connect(self.update_NeuralCrash)
        self.QObject_load_model.done.connect(self.threadAnim_load_model.requestInterruption)
        self.QObject_load_model.done.connect(self.check_function_load_testdata)
        self.threadAnim_load_model.time_passes.connect(lambda: self.update_label(self.label_load_model))

        self.threadAnim_load_model.start()
        self.threadWork_load_model.start()

    def check_function_load_testdata(self):
        self.threadAnim_load_testdata = CheckingAnimation()
        self.threadWork_load_testdata = QtCore.QThread()
        self.QObject_load_testdata = Check_function(self.neuralconfig, 'load_testdata')
        self.QObject_load_testdata.moveToThread(self.threadWork_load_testdata)

        self.threadWork_load_testdata.started.connect(self.QObject_load_testdata.run)
        self.QObject_load_testdata.function_checked_signal.connect(self.update_status)
        self.QObject_load_testdata.error_signal.connect(self.errorprint)
        self.QObject_load_testdata.return_new_state.connect(self.update_NeuralCrash)
        self.QObject_load_testdata.done.connect(self.threadAnim_load_testdata.requestInterruption)
        self.QObject_load_testdata.done.connect(self.check_function_get_tested_values)
        self.threadAnim_load_testdata.time_passes.connect(lambda: self.update_label(self.label_load_testdata))

        self.threadAnim_load_testdata.start()
        self.threadWork_load_testdata.start()

    def check_function_get_tested_values(self):
        self.threadAnim_get_tested_values = CheckingAnimation()
        self.threadWork_get_tested_values = QtCore.QThread()
        self.QObject_get_tested_values = Check_function(self.neuralconfig, 'get_tested_values')
        self.QObject_get_tested_values.moveToThread(self.threadWork_get_tested_values)

        self.threadWork_get_tested_values.started.connect(self.QObject_get_tested_values.run)
        self.QObject_get_tested_values.function_checked_signal.connect(self.update_status)
        self.QObject_get_tested_values.error_signal.connect(self.errorprint)
        self.QObject_get_tested_values.return_new_state.connect(self.update_NeuralCrash)
        self.QObject_get_tested_values.done.connect(self.threadAnim_get_tested_values.requestInterruption)
        self.QObject_get_tested_values.done.connect(self.check_function_set_tested_values)
        self.threadAnim_get_tested_values.time_passes.connect(lambda: self.update_label(self.label_get_tested_values))

        self.threadAnim_get_tested_values.start()
        self.threadWork_get_tested_values.start()

    def check_function_set_tested_values(self):
        self.threadAnim_set_tested_values = CheckingAnimation()
        self.threadWork_set_tested_values = QtCore.QThread()
        self.QObject_set_tested_values = Check_function(self.neuralconfig, 'set_tested_values')
        self.QObject_set_tested_values.moveToThread(self.threadWork_set_tested_values)

        self.threadWork_set_tested_values.started.connect(self.QObject_set_tested_values.run)
        self.QObject_set_tested_values.function_checked_signal.connect(self.update_status)
        self.QObject_set_tested_values.error_signal.connect(self.errorprint)
        self.QObject_set_tested_values.done.connect(self.threadAnim_set_tested_values.requestInterruption)
        self.QObject_set_tested_values.done.connect(self.check_function_test_model)
        self.threadAnim_set_tested_values.time_passes.connect(lambda: self.update_label(self.label_set_tested_values))

        self.threadAnim_set_tested_values.start()
        self.threadWork_set_tested_values.start()

    def check_function_test_model(self):
        self.threadAnim_test_model = CheckingAnimation()
        self.threadWork_test_model = QtCore.QThread()
        self.QObject_test_model = Check_function(self.neuralconfig, 'test_model')
        self.QObject_test_model.moveToThread(self.threadWork_test_model)

        self.threadWork_test_model.started.connect(self.QObject_test_model.run)
        self.QObject_test_model.function_checked_signal.connect(self.update_status)
        self.QObject_test_model.error_signal.connect(self.errorprint)
        self.QObject_test_model.done.connect(self.threadAnim_test_model.requestInterruption)
        self.QObject_test_model.done.connect(self.end_checking)
        self.threadAnim_test_model.time_passes.connect(lambda: self.update_label(self.label_test_model))

        self.threadAnim_test_model.start()
        self.threadWork_test_model.start()

    def end_checking(self):
        status_tuple = (
            self.status_load_model,
            self.status_load_testdata,
            self.status_get_tested_values,
            self.status_set_tested_values,
            self.status_test_model,
        )
        for status in status_tuple:
            if status != 'OK':
                return     
        self.load_config_label()
    

class CheckingAnimation(QtCore.QThread):
    time_passes = QtCore.pyqtSignal()
    def __init__(self):
        QtCore.QThread.__init__(self)

    def run(self):
        while not self.isInterruptionRequested():
            self.time_passes.emit()
            time.sleep(0.5)

class Check_function(QtCore.QObject):
    return_new_state = QtCore.pyqtSignal(NeuralCrach)
    function_checked_signal = QtCore.pyqtSignal(str, str)
    error_signal = QtCore.pyqtSignal(str, str)
    done = QtCore.pyqtSignal()

    def __init__(self, neuralconfig, name):
        QtCore.QObject.__init__(self)
        self.neuralconfig = neuralconfig
        self.name = name

    def run(self):
        try:
            if self.name == 'load_model':
                self.neuralconfig.load_model()

            elif self.name == 'load_testdata':
                self.neuralconfig.load_testdata()

            elif self.name == 'get_tested_values':
                self.neuralconfig._basedata = self.neuralconfig.get_tested_values()

            elif self.name == 'set_tested_values':
                self.neuralconfig.set_tested_values(self.neuralconfig._basedata)
                del(self.neuralconfig._basedata)

            elif self.name == 'test_model':
                self.neuralconfig.test_model(self.neuralconfig.model)

            self.function_checked_signal.emit(self.name, 'OK')    
        except Exception as e:
            self.function_checked_signal.emit(self.name, 'ERROR')
            self.error_signal.emit(str(e), self.name)

        self.return_new_state.emit(self.neuralconfig)    
        self.done.emit()      

class _NeuralCrashWindow(QMainWindow, _NeuralCrashWindowUI):
    def __init__(self, neuralconfig, parent=None):
        super(_NeuralCrashWindow, self).__init__(parent) 
        self.setupUI(self, neuralconfig)