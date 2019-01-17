from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
class Generator(BaseGenerator):

    def __init__(self, config):
        super(Generator, self).__init__(sess, config)

    def generate_random_X(shape):

        assert(shape[1] == 2), "Supports only num_agent = 2"  
    	X = np.zeros(shape)
        X[:, 0, :] = np.random.uniform(1.0, 2.0, size = (shape[0], shape[2]))
        X[:, 1, :] = np.random.uniform(1.0, 5.0, size = (shape[0], shape[2]))
        return X

    def generate_random_ADV(shape):

        assert(shape[2] == 2), "Supports only num_agent = 2"
        ADV = np.zeros(shape)  
        ADV[:, :, 0, :] = np.random.uniform(1.0, 2.0, size = (shape[0], shape[1], shape[3]))
        ADV[:, :, 1, :] = np.random.uniform(1.0, 5.0, size = (shape[0], shape[1], shape[3]))