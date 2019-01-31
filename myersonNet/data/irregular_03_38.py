from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from base.base_generator import BaseGenerator

class Generator(BaseGenerator):

    def __init__(self, config, mode, X = None):
        super(Generator, self).__init__(config, mode)
        self.build_generator(X = X)

    def generate_random_X(self, shape):
        # x is drawn from U[0, 3] with probability 0.75 and from U[3, 8] with probability 0.25
        r = np.random.rand(*shape)          
        X = np.zeros(shape)
        X[r <= 0.75] = np.random.uniform(0.0, 3.0, size = X[r <= 0.75].shape)
        X[r > 0.75] = np.random.uniform(3.0, 8.0, size = X[r > 0.75].shape)
        return X
