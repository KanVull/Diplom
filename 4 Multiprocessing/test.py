import PyQt5
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
from tensorflow.keras.datasets import fashion_mnist
import numpy as np
import time
import multiprocessing
from functools import partial
import pickle
import copy
from PyQt5 import QtCore, QtGui

from tensorflow.keras.models import Model
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils

def unpack(model, training_config, weights):
    restored_model = deserialize(model)
    if training_config is not None:
        restored_model.compile(
            **saving_utils.compile_args_from_training_config(
                training_config
            )
        )
    restored_model.set_weights(weights)
    return restored_model
# Hotfix function
def make_keras_picklable():

    def __reduce__(self):
        model_metadata = saving_utils.model_metadata(self)
        training_config = model_metadata.get("training_config", None)
        model = serialize(self)
        weights = self.get_weights()
        return (unpack, (model, training_config, weights))

    cls = Model
    cls.__reduce__ = __reduce__
# Run the function
make_keras_picklable()

class Data():
    model = None
    signal = QtCore.pyqtSignal(int)
    def load_model(self):
        self.model = models.load_model('../2 II Creation/fashion_mnist.h5')   

    def load_testdata(self):
        (_, _), (testX, testY) = fashion_mnist.load_data()
        testX = testX.reshape(testX.shape[0], 784) / 255
        self.testdata = (testX, testY)

    def get_tested_values(self):
        return self.model.get_weights()

    def set_tested_values(self, values):
        new_model = copy.deepcopy(self.model)
        new_model.set_weights(values)
        return new_model

    def test_model(self, model, data):
        results = model.evaluate(data[0], data[1], batch_size=64)
        return results[-1]               

    def _random_numpy_normal(self, values: list, sigma):
        return [np.random.normal(array, sigma) for array in values]    

    def _create_new_model(self, sigma, weights):
        weights = self._random_numpy_normal(weights, sigma)
        return self.set_tested_values(weights)

    def _compute_models(self, sigmas):
        data = list()
        weights = self.get_tested_values()
        all_have_to_do = int(sigmas[1]) -  int(sigmas[0])
        for index, sigma in enumerate(range(int(sigmas[0]), int(sigmas[1]))):
            data.append((sigma / 10000, self.test_model(self._create_new_model(sigma / 10000, weights), self.testdata)))
            self.signal.emit(all_have_to_do - index)
        return data  

    def multiprocessingCalc(self, sigma):
        cpu_count = multiprocessing.cpu_count() - 1
        sigma *= 10000
        sigma_step = int(sigma / cpu_count)
        list_in_data = list()
        sigma_start = 0
        while sigma_start < sigma:
            sigma_end = sigma_start + sigma_step
            if sigma_end > sigma - sigma_step:
                sigma_end = sigma + 1
            list_in_data.append((sigma_start, sigma_end))
            if sigma_end >= sigma:
                break
            sigma_start += sigma_step

        self.signal.connect(lambda: print('jija'))

        if self.model is not None:
            with multiprocessing.Pool(cpu_count) as p:
                data = p.map_async(self._compute_models, list_in_data)
                data = data.get()
                # while (True):
                #     if (data.ready()): break
                #     remaining = data._number_left
                #     print ("Waiting for", remaining, "tasks to complete...")
                #     time.sleep(0.5)
        else:
            data = None
            print('Model not loaded')
        print(data)


if __name__ == '__main__':
    SIGMA = 0.05

    data = Data()
    data.load_model()
    data.load_testdata()
    t = time.time()
    data.multiprocessingCalc(SIGMA)
    print(time.time() - t)
    # print(data)
    