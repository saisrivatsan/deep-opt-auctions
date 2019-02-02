from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
cfg = __C

# Output-dir to write log-files and save model
__C.dir_name = os.path.join("experiments", "m_5x1_asym_uniform")

# Auction params
__C.num_agents = 5
__C.distribution_type = "asymmetric_uniform"


# Save data for restore.
__C.save_data = False

""" Neural Net parameter """
__C.net = edict()    
# Init
__C.net.b_init = [-5.0, 0.0]
# Num max units
__C.net.num_max_units = 10
# Num func
__C.net.num_func = 10
# soft-max constant for smooth argmax approximation
__C.net.eps = 1e3


""" Train paramters """
__C.train = edict()

# Random seed
__C.train.seed = 42
# Iter from which training begins. If restore_iter = 0 for default. restore_iter > 0 for starting
# training form restore_iter [needs saved model]
__C.train.restore_iter = 0
# max iters to train 
__C.train.max_iter = 40000
# Learning rate of network param updates
__C.train.learning_rate = 1e-3
# Regularization
__C.train.wd = None


""" Train-data params """
# Choose between fixed and online.
__C.train.data = "fixed"
# Number of batches
__C.train.num_batches = 5000
# Train batch size
__C.train.batch_size = 128


""" train summary and save params"""
# Number of models to store on disk
__C.train.max_to_keep = 4
# Frequency at which models are saved
__C.train.save_iter = 40000
# Train stats print frequency
__C.train.print_iter = 10000
   

""" Validation params """
__C.val = edict()
# Number of batches
__C.val.num_batches = 100
# Frequency at which validation is performed
__C.val.print_iter = 10000
# Validation data
__C.val.data = "fixed"

""" Test params """
# Test set
__C.test = edict()
# Test Seed
__C.test.seed = 100
# Model to be evaluated
__C.test.restore_iter = 40000
# Test data
__C.test.data = "online"
# Number of test batches
__C.test.num_batches = 100
# Test batch size
__C.test.batch_size = 100
# Save Ouput
__C.test.save_output = False


# Fixed Val params
__C.val.batch_size = __C.train.batch_size

# Compute number of samples
__C.train.num_instances = __C.train.num_batches * __C.train.batch_size
__C.val.num_instances = __C.val.num_batches * __C.val.batch_size
__C.test.num_instances = __C.test.num_batches * __C.test.batch_size

