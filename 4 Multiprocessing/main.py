import multiprocessing
from multiprocessing import shared_memory
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model, load_model
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils
import numpy as np
import time

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
        shm = shared_memory.SharedMemory(create=True, size=testX.nbytes)
        self.X = np.ndarray(testX.shape, dtype=testX.dtype, buffer=shm.buf)
        # self.X = testX
        self.Y = testY

    def get_values_from_model(self, model):
        return model.get_weights()

    def set_values_to_model(self, values, model):
        model.set_weights(values)
        return model

    def evaluate_model(self, model):
        results = model.evaluate(self.X, self.Y, batch_size=64)
        return results[-1] 

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
            data.append((sigma / 10000, self.evaluate_model(self._create_new_model(sigma / 10000))))
        return data  


    def run(self, sigma):
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

        if self.model is not None:
            with multiprocessing.Pool(cpu_count) as p:
                data = p.map(self._compute_models, list_in_data)
            # data = self._compute_models((0, 5))
        else:
            data = None
            print('Model not loaded')
        print(data)   

    def stupid_run(self, sigma):
        sigma *= 10000

        if self.model is not None:
            data = list()
            for s in range(int(sigma)+1):
                data.append(self._compute_models((s,s+1)))
        else:
            data = None
            print('Model not loaded')
        print(data)                    

if __name__ == '__main__':
    SIGMA = 0.005
    work = NeuralCrashTest()
    work.load_model('../2 II Creation/fashion_mnist.h5')
    work.load_test_data()
    now = time.time()
    work.run(SIGMA)
    print(time.time() - now)
    now = time.time()
    work.stupid_run(SIGMA)
    print(time.time() - now)
                        