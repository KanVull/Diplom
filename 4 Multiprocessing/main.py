import multiprocessing
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import fashion_mnist

import numpy as np

class NeuralCrashTest():
    def __init__(self):
        self.model = None

    def load_model(self, path):
        try:
            self.model = load_model(path)
        except ImportError:
            return False
        except IOError:
            return False
        return True    

    def load_test_data(self):
        (_, _), (testX, testY) = fashion_mnist.load_data()
        del(_)
        testX = testX.reshape(testX.shape[0], 784) / 255
        self.X = testX
        self.Y = testY

    def get_values_from_model(self, model):
        for variable in model.weights:
            if type(variable) == tensorflow.python.obs.ResourceVariable:
                values = np.array(variable)
        return values  

    def set_values_to_model(self, values, model):
        for variable in model.weights:
            if type(variable) == tensorflow.python.obs.ResourceVariable:
                variable.assign(values)
        return model

    def _random_numpy_normal(self, values: np.array, sigma):
        return np.random.normal(values, sigma)

    def run(self):
        pass                