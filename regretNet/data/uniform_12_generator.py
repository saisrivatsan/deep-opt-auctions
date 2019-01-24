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
        return np.random.rand(*shape) + 1.0

    def generate_random_ADV(self, shape):
        return np.random.rand(*shape) + 1.0