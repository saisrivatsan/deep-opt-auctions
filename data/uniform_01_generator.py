from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
class Generator(BaseGenerator):

    def __init__(self, config):
        super(Generator, self).__init__(sess, config)

    def generate_random_X(shape):
        return np.random.rand(*shape)

    def generate_random_ADV(shape):
        return np.random.rand(*shape)