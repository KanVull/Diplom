from tensorflow import keras
from tensorflow.keras import models
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
from tensorflow.keras.datasets import fashion_mnist
import numpy as np
import time
import multiprocessing
from functools import partial

def new_model(sigma, model):
        model = model
        for variable in model.weights:
            if type(variable) == ResourceVariable:
                arr = np.array(variable)
                new_weights = np.random.normal(arr, sigma/10000)
                variable.assign(new_weights)
        results = model.evaluate(testX, testY, batch_size=64)           
        return results    

def compute_models(model, sigmas):
    data = list()
    for sigma in range(int(sigmas[0]), int(sigmas[1])):
        data.append(new_model(sigma, model))
    return data   

def multiprocessingCalc(sigma, model):
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

    with multiprocessing.Pool(cpu_count) as p:
        data = p.map(partial(compute_models, model=model), list_in_data)

    return data


if __name__ == '__main__':
    SIGMA = 0.4
    model = keras.models.load_model('../2 II Creation/fashion_mnist.h5')
    (_, _), (testX, testY) = fashion_mnist.load_data()
    del(_)
    testX = testX.reshape(testX.shape[0], 784) / 255
    
    model_copy = model
    print("Evaluate on test data")
    true_results = model.evaluate(testX, testY, batch_size=64)
    print("test loss, test acc:", true_results, '\n\n')

    # happy_moments = 0
    # best_solution = None
    # sad_moments = 0

    # model_copy = model
    # start_time = time.time()
    # for sigma in range(int(SIGMA*10000)):
    #     results = new_model(sigma, model_copy)
    #     if results[-1] >= true_results[-1]:
    #         if best_solution is not None:
    #             if results[-1] > best_solution[0]:
    #                 best_solution[0] = results[-1]
    #                 best_solution[1] = model_copy
    #         else:
    #             best_solution = [results[-1], model_copy]
    #         happy_moments += 1
    #     else:
    #         sad_moments += 1               
    #     print("test loss, test acc:", results)

    # print(f'\nHappy_moments: {happy_moments}')
    # if happy_moments > 0:
    #     print(f'We have a better model with accuracy: {best_solution[0]}\nYour model have {true_results[1]} accuracy')
    # print(f'Sad moments: {sad_moments}')    
    # print("--- LOOP STUPID METHOD %s seconds ---" % (time.time() - start_time))

    data = multiprocessingCalc(SIGMA, model_copy)
    print(data)
    