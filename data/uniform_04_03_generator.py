from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from base.base_generator import BaseGenerator

class Generator(BaseGenerator):

    def __init__(self, config, mode, X = None, ADV = None):
        super(Generator, self).__init__(config, mode)
        self.build_generator(X = X, ADV = ADV)

    def generate_random_X(self, shape):
        X = np.zeros(shape)
        size = (shape[0], shape[1])
        X[:, :, 0] = np.random.uniform(0.0, 4.0, size = size)
        X[:, :, 1] = np.random.uniform(0.0, 3.0, size = size)
        return X

    def generate_random_ADV(self, shape):
        X = np.zeros(shape)
        size = (shape[0], shape[1], shape[2])
        X[:, :, :, 0] = np.random.uniform(0.0, 4.0, size = size)
        X[:, :, :, 1] = np.random.uniform(0.0, 3.0, size = size)
        return X
    
