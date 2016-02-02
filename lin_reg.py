import numpy as np
import pylab as pl

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

training_data = np.loadtxt('data-train.csv', delimiter=',')
test_data = np.loadtxt('data-test.csv', delimiter=',')

x = training_data[:, :2]
y = training_data[:, 2]
x_test = test_data[:, :2]
y_test = test_data[:, 2]

size = y.size
test_size = y_test.size
iterations = 87
learning_rate = 0.1
weights = np.ones(shape=(2,1)).flatten()
bias = 1

error_array = []
error_test_array = []
for i in range(iterations):
    weights, bias = gradient_descent(x, y, weights, bias, learning_rate, size)
    error = mean_squared_error(x, y, weights, bias, size)
    error_array.append(error)
    error_test = mean_squared_error(x_test, y_test, weights, bias, test_size)
    error_test_array.append(error_test)

print("weight",weights)
print("bias",bias)
print("error", error)
print("test error", error_test)
pl.plot(pl.arange(iterations), error_array)
pl.xlabel('Iterations')
pl.ylabel('Mean Squared Error (training)')
pl.show()
