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
__C.dir_name = os.path.join("experiments", "unit_1x2_uniform")

# Auction params
__C.num_agents = 1
__C.num_items = 2
__C.distribution_type = "uniform"
__C.agent_type = "unit_demand"

# Save data for restore.
__C.save_data = False

# Neural Net parameters
__C.net = edict()    
# initialization g - glorot, h - he + u - uniform, n - normal [gu, gn, hu, hn]
__C.net.init = "gu"
# activations ["tanh", "sigmoid", "relu"]
__C.net.activation = "tanh"
# num_a_layers, num_p_layers - total number of hidden_layers + output_layer, [a - alloc, p - pay]
__C.net.num_a_layers = 3
__C.net.num_p_layers = 3
# num_p_hidden_units, num_p_hidden_units - number of hidden units, [a - alloc, p - pay]
__C.net.num_p_hidden_units = 100
__C.net.num_a_hidden_units = 100

# Train paramters
__C.train = edict()

# Random seed
__C.train.seed = 42
# Iter from which training begins. If restore_iter = 0 for default. restore_iter > 0 for starting
# training form restore_iter [needs saved model]
__C.train.restore_iter = 0
# max iters to train 
__C.train.max_iter = 400000
# Learning rate of network param updates
__C.train.learning_rate = 1e-3
# Regularization
__C.train.wd = None

""" Train-data params """
# Choose between fixed and online. If online, set adv_reuse to False
__C.train.data = "fixed"
# Number of batches
__C.train.num_batches = 5000
# Train batch size
__C.train.batch_size = 128


""" Train-misreport params """
# Cache-misreports after misreport optimization
__C.train.adv_reuse = True
# Number of misreport initialization for training
__C.train.num_misreports = 1
# Number of steps for misreport computation
__C.train.gd_iter = 25
# Learning rate of misreport computation
__C.train.gd_lr = 0.1

""" Lagrange Optimization params """
# Initial update rate
__C.train.update_rate = 1.0
# Initial Lagrange weights
__C.train.w_rgt_init_val = 5.0
# Lagrange update frequency
__C.train.update_frequency = 100
# Value by which update rate is incremented
__C.train.up_op_add = 20.0
# Frequency at which update rate is incremented
__C.train.up_op_frequency = 10000


""" train summary and save params"""
# Number of models to store on disk
__C.train.max_to_keep = 10
# Frequency at which models are saved
__C.train.save_iter = 50000 
# Train stats print frequency
__C.train.print_iter = 1000
   

""" Validation params """
__C.val = edict()
# Number of steps for misreport computation
__C.val.gd_iter = 2000
# Learning rate for misreport computation
__C.val.gd_lr = 0.1
# Number of validation batches
__C.val.num_batches = 20
# Frequency at which validation is performed
__C.val.print_iter = 10000
# Validation data frequency
__C.val.data = "fixed"

""" Test params """
# Test set
__C.test = edict()
# Test Seed
__C.test.seed = 100
# Model to be evaluated
__C.test.restore_iter = 400000
# Number of misreports
__C.test.num_misreports = 1000
# Number of steps for misreport computation
__C.test.gd_iter = 2000
# Learning rate for misreport computation
__C.test.gd_lr = 0.1
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
__C.val.num_misreports = __C.train.num_misreports

# Compute number of samples
__C.train.num_instances = __C.train.num_batches * __C.train.batch_size
__C.val.num_instances = __C.val.num_batches * __C.val.batch_size
__C.test.num_instances = __C.test.num_batches * __C.test.batch_size

