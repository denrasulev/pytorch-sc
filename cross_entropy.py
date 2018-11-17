# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.

import numpy as np


def cross_entropy(Y, P):
    Y = np.float64(Y)
    P = np.float64(P)
    return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))