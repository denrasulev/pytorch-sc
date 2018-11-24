# SoftMax function

import torch
import numpy as np


# SoftmMax in NumPy
def softmax1(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


# SoftMax in PyTorch
def softmax(x):
    return torch.exp(x) / torch.sum(torch.exp(x))
