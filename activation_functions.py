import numpy as np


# Sigmoid
def sigmoid(x, derivative=False):
    if derivative is True:
        return x * (1 - x)
    else:
        return 1 / (1 + np.exp(-x))


# Hyperbolic Tangent Function
def tanh(x, derivative=False):
    if derivative is True:
        return 1 - x ** 2
    else:
        return np.tanh(x)


# ReLU - Rectified Linear Unit
def ReLU(x):
    return x * (x > 0)


# Sigmoid - avoid in practice!
# Sigmoids saturate and kill gradients
# Sigmoids slow confergence
# Sigmoids are not zero centered
# OK to use on last layer

# TanH - avoid in practice!

# ReLU - Rectified Linear Unit
# Learns faster
# Avoids vanishing gradient
# Only use for the hidden layers!
# ReLU could result in dead neurons (then use Leaked ReLU)
# Output - Use SoftMax (or Sigmoid) for Multi Class
# Output - Use Linear (or ReLU) function for regression
# More implementations:
# https://stackoverflow.com/questions/32109319/how-to-implement-the-relu-function-in-numpy
