# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.

import numpy as np


def softmax(L):
    return([np.exp(n) / sum([np.exp(i) for i in L]) for n in L])
