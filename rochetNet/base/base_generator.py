from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np


class BaseGenerator(object):
    def __init__(self, config, mode = "train"):    
        self.config = config
        self.mode = mode
        self.num_items = config.num_items
        self.num_instances = config[self.mode].num_batches * config[self.mode].batch_size
        self.batch_size = config[self.mode].batch_size
                       
    def build_generator(self, X = None):
        if self.mode is "train":            
            if self.config.train.data is "fixed":
                if self.config.train.restore_iter == 0:
                    self.get_data(X)
                else:
                    self.load_data_from_file()
                self.gen_func = self.gen_fixed()
            else:
                self.gen_func = self.gen_online()
                
        else:
            if self.config[self.mode].data is "fixed" or X is not None:
                self.get_data(X)
                self.gen_func = self.gen_fixed()
            else:
                self.gen_func = self.gen_online()
            

        
    def get_data(self, X = None):
        """ Generates data """
        x_shape = [self.num_instances, self.num_items]
        
        if X is None: X = self.generate_random_X(x_shape)
        self.X = X
                       
    def load_data_from_file(self):
        """ Loads data from disk """
        self.X = np.load(os.path.join(self.config.dir_name, 'X.npy'))

    def save_data(self):
        """ Saves data to disk """
        if self.config.save_data is False: return
        np.save(os.path.join(self.config.dir_name, 'X'), self.X)
                       
    def gen_fixed(self):
        i = 0
        if self.mode is "train": perm = np.random.permutation(self.num_instances) 
        else: perm = np.arange(self.num_instances)
        while True:
            idx = perm[i * self.batch_size: (i + 1) * self.batch_size]
            yield self.X[idx]
            i += 1
            if(i * self.batch_size == self.num_instances):
                i = 0
                if self.mode is "train": perm = np.random.permutation(self.num_instances) 
                else: perm = np.arange(self.num_instances)
            
    def gen_online(self):
        x_batch_shape = [self.batch_size, self.num_items]
        while True:
            X = self.generate_random_X(x_batch_shape)
            yield X

    def generate_random_X(self, shape):
        """ Rewrite this for new distributions """
        raise NotImplementedError
