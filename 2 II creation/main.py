from tensorflow import keras, nn
from tensorflow.keras.datasets import fashion_mnist
import math
import numpy as np

NUM_EPOCHS = 25
BS = 64

(trainX, trainY), (testX, testY) = fashion_mnist.load_data()

trainX = trainX.reshape(trainX.shape[0], 784) / 255
testX = testX.reshape(testX.shape[0], 784) / 255

model = keras.Sequential([
    keras.layers.Dense(800, input_dim=28*28, activation=nn.relu),
    keras.layers.Dense(10, activation=nn.softmax)
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

model.save('fashion_mnist.h5')

print('h')