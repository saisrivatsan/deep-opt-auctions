from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from random import random

class Generator():
    
    def __init__(self, args, seed_val):
        self.args = args
        self.seed_val = seed_val
        
    def generate_sample(self, str):
        num_agent = self.args.num_agent
        num_item = self.args.num_item
        if str == 'train':
            num_instances = self.args.num_sample_train
        elif str == 'test':
            num_instances = self.args.num_sample_test

        distr_type = self.args.distribution_type
        
        
        np.random.seed(self.seed_val)
                                 
        if distr_type == 'uniform':
            sample_val = np.random.rand(num_instances, num_agent)
          
        elif distr_type == 'irregular':
            sample_val = np.zeros((num_instances, num_agent))
            for i in range(num_instances):
                for j in range(num_agent):
                    if random() < 0.75:
                        sample_val[i, j] = np.random.uniform(0,3)
                    else:
                        sample_val[i, j] = np.random.uniform(3,8)
                        
        elif distr_type == 'exponential':
            sample_val = np.random.exponential(3.0, size=(num_instances, num_agent))
            
        elif distr_type == 'asymmetric_uniform':
            sample_val = np.zeros((num_instances, num_agent))
            for i in range(num_agent):
                sample_val[:,i] = np.float32(np.random.rand(num_instances) * (i+1))
                
        return sample_val
            
        
           