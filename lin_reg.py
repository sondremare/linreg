#from numpy import loadtxt, zeros, ones, array, linspace, logspace, mean, std, arange, insert
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from pylab import plot, show, xlabel, ylabel

def prediction(x, weights, bias):
    return (x.dot(weights) + bias).flatten()

def mean_squared_error(x, y, weights, bias, size):
    predicted_value = prediction(x, weights, bias)
    return ((1/size)*(predicted_value - y)**2).sum()


def gradient_descent(x, y, weights, bias, learning_rate, size):
    weights_deriv = (2 / size) * (prediction(x, weights, bias) - y).dot(x)
    bias_deriv = (2 / size) * (prediction(x, weights, bias) - y)
    weights = weights - (learning_rate * weights_deriv)
    bias = bias - (learning_rate * bias_deriv.sum())
    return weights, bias

#Load the datasets
#training_data = np.loadtxt('data-train.csv', delimiter=',')
training_data = np.loadtxt('data-train.csv', delimiter=',')
test_data = np.loadtxt('data-test.csv', delimiter=',')

#Splitting training data into input vector x and target output y
x = training_data[:, :2]
y = training_data[:, 2]

size = y.size
#Adding intercept term
intercept_terms = np.ones(shape=(size,1))
#x = np.append(intercept_terms, x, axis=1)

iterations = 100
learning_rate = 0.01

weights = np.zeros(shape=(2,1)).flatten()
bias = 0;

for i in range(iterations):
    weights, bias = gradient_descent(x, y, weights, bias, learning_rate, size)
    error = mean_squared_error(x,y,weights,bias,size)
    print("e",error)
    #print("w",weights)
    #print("b",bias)

#print(weights)