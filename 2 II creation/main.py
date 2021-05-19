from tensorflow import keras, nn
from tensorflow.keras.datasets import fashion_mnist
import math
import numpy as np

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
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

NUM_EPOCHS = 25
BS = 64

(trainX, trainY), (testX, testY) = fashion_mnist.load_data()

trainX = trainX.reshape(trainX.shape[0], 784) / 255
testX = testX.reshape(testX.shape[0], 784) / 255

model = Sequential([
    Dense(800, input_dim=28*28, activation=nn.relu),
    Dense(10, activation=nn.softmax)
])
model.compile(optimizer="SGD",
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
)
model.fit(trainX, trainY, 
          epochs=NUM_EPOCHS, 
          batch_size=BS,
          verbose=1,
          validation_data=(testX, testY)
)

print("Evaluate on test data")
results = model.evaluate(testX, testY, batch_size=BS)
print("test loss, test acc:", results)

model.save('fashion_mnist_try_pickle.h5')

print('h')