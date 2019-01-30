from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from base.base_generator import BaseGenerator

class Generator(BaseGenerator):
    def __init__(self, config, mode, X = None, ADV = None):
        super(Generator, self).__init__(config, mode)
        self.build_generator(X = X)
        assert(self.config.num_items == 2)

    def generate_random_X(self, shape):
        r = np.random.rand(*shape)
        x = np.zeros(shape)
        x[:, 0] = np.sqrt(r[:, 0] )* (1 - r[:, 1])
        x[:, 1] = np.sqrt(r[:, 0]) * (r[:, 1])
        return x
