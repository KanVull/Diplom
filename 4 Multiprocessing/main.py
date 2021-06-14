from NeuralCrash import NeuralCrach
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model, load_model
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils
import copy

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
def make_keras_picklable():

    def __reduce__(self):
        model_metadata = saving_utils.model_metadata(self)
        training_config = model_metadata.get("training_config", None)
        model = serialize(self)
        weights = self.get_weights()
        return (unpack, (model, training_config, weights))

    cls = Model
    cls.__reduce__ = __reduce__
make_keras_picklable()

class NeuralCrashTest(NeuralCrach):
    def __init__(self):
        super().__init__()

    def load_model(self):
        self.model = load_model('../2 II Creation/fashion_mnist.h5')   

    def load_testdata(self):
        (_, _), (testX, testY) = fashion_mnist.load_data()
        testX = testX.reshape(testX.shape[0], 784) / 255
        self.testdata = testX, _

    def get_tested_values(self):
        return self.model.get_weights()

    def set_tested_values(self, values):
        new_model = self.model
        new_model.set_weights(values)
        return new_model

    def test_model(self, model, data):
        results = model.evaluate(data[0], data[1], batch_size=64)
        return results[-1]                    

if __name__ == '__main__':
    work = NeuralCrashTest()
    work.setName('Tensorflow model')

    work.run()
                        