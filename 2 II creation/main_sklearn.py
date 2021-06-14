import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

digits = datasets.load_digits()

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
clf = svm.SVC(gamma=0.001)

# Split data into 50% train and 50% test subsets
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False)

# Learn the digits on the train subset
clf.fit(X_train, y_train)

predicted = clf.predict(X_test)

print(accuracy_score(y_test, predicted))

print(clf._get_coef())

clf.dual_coef_ = np.array([np.random.normal(array, 0.4) for array in clf._get_coef()])
# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)

print(accuracy_score(y_test, predicted))

print(clf._get_coef())

          