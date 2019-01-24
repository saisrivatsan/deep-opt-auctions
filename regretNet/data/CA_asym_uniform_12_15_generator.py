from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from base.base_generator_ca import BaseGenerator

class Generator(BaseGenerator):

    def __init__(self, config, mode, X = None, ADV = None):
        super(Generator, self).__init__(config, mode)
        self.build_generator(X = X, ADV = ADV)

    def generate_random_X(self, shape):
        X = np.zeros(shape)
        size = (shape[0], shape[2])
        X[:, 0, :] = np.random.uniform(1.0, 2.0, size = size)
        X[:, 1, :] = np.random.uniform(1.0, 5.0, size = size)
        return X

    def generate_random_ADV(self, shape):
        X = np.zeros(shape)
        size = (shape[0], shape[1], shape[3])
        X[:, :, 0, :] = np.random.uniform(1.0, 2.0, size = size)
        X[:, :, 1, :] = np.random.uniform(1.0, 5.0, size = size)
        return X
    
    def generate_random_C(self, shape):
        return np.random.uniform(-1.0, 1.0, size = shape)