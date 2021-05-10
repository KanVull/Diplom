from tensorflow import keras
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
from tensorflow.keras.datasets import fashion_mnist
import numpy as np
import time
import multiprocessing

def new_model(model, sigma):
    for index, variable in enumerate(model.weights):
        if type(variable) == ResourceVariable:
            arr = np.array(variable)
            new_weights = np.random.normal(arr, sigma/10000)
            variable.assign(new_weights)
    print("Evaluate on test data modifyed", sigma/10000)
    results = model.evaluate(testX, testY, batch_size=64)           
    return results

def poolmethod(model):
    happy_moments = 0
    best_solution = None
    sad_moments = 0

    pool = multiprocessing.Pool()
    start_time = time.time()

    for x in pool.imap_unordered(new_model, [model_copy, range(int(SIGMA*1000))]):
        if x[-1] >= true_results[-1]:
            if best_solution is not None:
                if x[-1] > best_solution[0]:
                    best_solution[0] = x[-1]
                    best_solution[1] = model_copy
            else:
                best_solution = [x[-1], model_copy]
            happy_moments += 1
        else:
            sad_moments += 1    

    print(f'\nHappy_moments: {happy_moments}')
    if happy_moments > 0:
        print(f'We have a better model with accuracy: {best_solution[0]}\nYour model have {true_results[1]} accuracy')
    print(f'Sad moments: {sad_moments}')    
    print("--- WOW MULTIPROCESSING METHOD %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
    SIGMA = 0.05
    model = keras.models.load_model('../2 II Creation/fashion_mnist.h5')
    (_, _), (testX, testY) = fashion_mnist.load_data()
    testX = testX.reshape(testX.shape[0], 784) / 255
    print("Evaluate on test data")
    true_results = model.evaluate(testX, testY, batch_size=64)
    print("test loss, test acc:", true_results, '\n\n')

    happy_moments = 0
    best_solution = None
    sad_moments = 0

    model_copy = model
    start_time = time.time()
    for sigma in range(int(SIGMA*1000)):
        results = new_model(model_copy, sigma)
        if results[-1] >= true_results[-1]:
            if best_solution is not None:
                if results[-1] > best_solution[0]:
                    best_solution[0] = results[-1]
                    best_solution[1] = model_copy
            else:
                best_solution = [results[-1], model_copy]
            happy_moments += 1
        else:
            sad_moments += 1               
        print("test loss, test acc:", results)

    print(f'\nHappy_moments: {happy_moments}')
    if happy_moments > 0:
        print(f'We have a better model with accuracy: {best_solution[0]}\nYour model have {true_results[1]} accuracy')
    print(f'Sad moments: {sad_moments}')    
    print("--- LOOP STUPID METHOD %s seconds ---" % (time.time() - start_time))

    freeze_support()
    poolmethod(model_copy)
    