# Avoid in practice!
#
# Sigmoids saturate and kill gradients
# Sigmoids slow confergence
# Sigmoids are not zero centered
# OK to use on last layer

import numpy as np


def sigmoid(x, derivative=False):
    if derivative is True:
        return x * (1 - x)
    else:
        return 1 / (1 + np.exp(-x))
