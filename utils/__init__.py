import numpy as np

from .pgm import *
from .constants import *

# Numpy utils

def to_onehot(a, shape):
    if shape[0] is None:
        shape = len(a), shape[1]
    onehot = np.zeros(shape)
    onehot[np.arange(len(a)), a] = 1
    return onehot

