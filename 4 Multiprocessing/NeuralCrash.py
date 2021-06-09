import numpy as np
import multiprocessing
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QMainWindow, 
    QDialog, 
    QDialogButtonBox,
    QMessageBox,
    QSpinBox,
    QTextEdit,
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
        self.setnewcentralwidget('Checking override functions', self.create_checking_widget())

    def setnewcentralwidget(self, name, widget):
        self.clearLayout(self.centralWidgetLayout)
        label = QLabel()
        label.setText(name)
        self.centralWidgetLayout.addWidget(label)
        self.centralWidgetLayout.addWidget(widget)
        self.MainWindow.setCentralWidget(self.centralWidget)
        self.MainWindow.setContentsMargins(0, 0, 0, 0)

    def create_checking_widget(self):
        checkingWidget = CheckingWidget(self.neuralconfig)
        checkingWidget.checking_ending.connect(self.create_test_config_widget)
        return checkingWidget

    def create_test_config_widget(self, neuralconfig):
        self.neuralconfig = neuralconfig
        testConfigWidget = TestConfigWidget()
        self.setnewcentralwidget('Set test configuration', testConfigWidget)

class CheckingWidget_UI(object):
    checking_ending = QtCore.pyqtSignal(NeuralCrach)

    def setupUI(self, widget, neuralconfig):
        self.neuralconfig = neuralconfig
        self.layout = QVBoxLayout(widget)

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

        self.layout.addLayout(status)
        self.layout.addWidget(self.error)
        self.check_function_load_model()

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
        self.checking_ending.emit(self.neuralconfig)

class CheckingWidget(QWidget, CheckingWidget_UI):
    def __init__(self, neuralconfig, parent=None):
        super(CheckingWidget, self).__init__(parent) 
        self.setupUI(self, neuralconfig)


class TestConfigWidget_UI(object):
    config = [i/10000 for i in range(0,1001,20)]
    config_changed = QtCore.pyqtSignal()

    def setupUI(self, widget):
        self.layout = QVBoxLayout(widget)
        
        self.config_layout = QFormLayout()

        self.num_processors = QSpinBox()
        self.num_processors.setMinimum(1)
        self.num_processors.setMaximum(self.getMaximumProcessorCount())
        label_processors = QLabel()
        label_processors.setText('Specify the number of processors used')
        self.config_layout.addRow(self.num_processors, label_processors)

        algorithms = GeneratingRandomAlgorithms()
        self.algorithmComboBox = QComboBox()
        self.algorithmComboBox.addItems(algorithms.names)
        algorithmLabel = QLabel()
        algorithmLabel.setText('Choose algorithm')
        self.config_layout.addRow(self.algorithmComboBox, algorithmLabel)

        self.layout.addLayout(self.config_layout)

    ## sigma widget
        self.sigmalayout = QVBoxLayout()
        
        ## 0 - start; 1 - stop; 2 - step; 3 - The number of tests at each step
        layouts_edits = [QVBoxLayout() for _ in range(4)]

        labels_edits = [QLabel() for _ in range(4)]
        labels_edits[0].setText('Start')
        labels_edits[1].setText('Stop')
        labels_edits[2].setText('Step')
        labels_edits[3].setText('NTES')
        labels_edits[3].setToolTip('The number of tests at each step')

        self.line_edits = [QLineEdit() for _ in range(4)]
        for i in range(3):
            self.line_edits[i].setToolTip('Int or float number; . is delimiter')
        self.line_edits[3].setToolTip('Int number > 0')

        self.line_edits[0].setText('0')
        self.line_edits[1].setText('0.005')
        self.line_edits[2].setText('0.0001')
        for i in range(4):
            self.line_edits[i].textEdited.connect(self.update)
        self.line_edits[3].setText('1')    

        layout_sssc = QHBoxLayout()
        for i in range(4):
            layouts_edits[i].addWidget(labels_edits[i])
            layouts_edits[i].addWidget(self.line_edits[i])
            layout_sssc.addLayout(layouts_edits[i])

        self.preview = QLabel()
        self.preview.setText('jope')

        self.sigmalayout.addLayout(layout_sssc)
        self.sigmalayout.addWidget(self.preview)

        self.config_changed.connect(self.setConfig)
        self.layout.addLayout(self.sigmalayout)

    ## sigma ended

        self.buttonstart = QPushButton()
        self.buttonstart.clicked.connect(self.onStarted)
        self.setConfig()
        self.buttonstart.setText('Start test')
        self.layout.addWidget(self.buttonstart)  

        self.num_processors.valueChanged[int].connect(self.update) 

    def getMaximumProcessorCount(self):
        return multiprocessing.cpu_count()    

    def setError(self, text):
        self.preview.setText('Config error | ' + text)
        self.config = None
        self.config_changed.emit()

    def update(self):
        numbers = (edit.text() for edit in self.line_edits)
        try:
            start = float(numbers[0])
            stop = float(numbers[1])
            step = float(numbers[2])
            ntes = int(numbers[3])
        except Exception:
            self.setError('Wrong numbers')
            return
        self.line_edits[3] = ntes    
        if start > stop:
            self.setError('Start > Stop')
            return
        elif (stop - start) / step < 1:
            self.setError('Unavailable Step')
            return
        elif ntes < 1:
            self.setError('NTES must be over 0')
            return
        self.config = np.arange(start, stop, step)
        if ntes > 1:
            self.config = np.repeat(self.config, ntes)
        if len(self.config) > 8:
            self.preview.setText('[' + ', '.join(self.config[:3]) + ', ..., ' + ', '.join(self.config[-3:]) + ']')
        else:
            self.preview.setText('[' + ', '.join(self.config) + ']') 
        self.config_changed.emit()       

    def setConfig(self):
        self.buttonstart.setEnabled(self.config is not None)

    def onStarted(self):
        print(self.config)  

    

class TestConfigWidget(QWidget, TestConfigWidget_UI):
    def __init__(self, parent=None):
        super(TestConfigWidget, self).__init__(parent) 
        self.setupUI(self)


class GeneratingRandomAlgorithms():
    names = [
        'normal',
        'not_normal'
    ]


class TestingWidget_UI(object):
    def setupUI(self):
        pass

class TestingWidget(QWidget, TestingWidget_UI):
    def __init__(self, parent=None):
        super(TestingWidget, self).__init__(parent) 
        self.setupUI(self)


class ResultWidget_UI(object):
    def setupUI(self):
        pass

class ResultWidget(QWidget, ResultWidget_UI):
    def __init__(self, parent=None):
        super(ResultWidget, self).__init__(parent) 
        self.setupUI(self)


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