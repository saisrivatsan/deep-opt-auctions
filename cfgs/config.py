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

__C.dir_name = "3x10_l6_10k_up_add_1.0"
__C.num_agents = 3
__C.num_items = 10
__C.save_data = False

# Neural Net parameters
__C.net = edict()
__C.net.init = "gu"
__C.net.activation = "tanh"
__C.net.num_a_layers = 6
__C.net.num_p_layers = 6
__C.net.num_p_hidden_units = 100
__C.net.num_a_hidden_units = 100

# Train paramters
__C.train = edict()

__C.train.up_op_add = 1.0
__C.train.up_op_frequency = 10000
__C.train.seed = 42
__C.train.restore_iter = 0
__C.train.num_misreports = 1
__C.train.batch_size = 128
__C.train.update_rate = 1.0
__C.train.learning_rate = 1e-3
__C.train.gd_lr = 0.1
__C.train.update_frequency = 100
__C.train.adv_reuse = True
__C.train.iter_1 = 1
__C.train.w_rgt_init_val = 5.0

__C.train.gd_iter = 25
__C.train.max_iter = 400000
__C.train.max_to_keep = 10
__C.train.data = "fixed"
__C.train.num_batches = 5000
__C.train.print_iter = 1000
__C.train.save_iter = 50000    

#__C.train.reg_const = 1e-3

# Validation set
__C.val = edict()
__C.val.gd_iter = 2000
__C.val.gd_lr = 0.1
__C.val.num_batches = 20
__C.val.print_iter = 10000
__C.val.data = "fixed"

# Test set
__C.test = edict()
__C.test.gd_iter = 2000
__C.test.gd_lr = 0.1
__C.test.num_batches = 50
__C.test.batch_size = 1024
__C.test.seed = 0
__C.test.restore_iter = 400000
__C.test.num_misreports = 1
__C.test.data = "online"

# Compute
__C.val.batch_size = __C.train.batch_size
__C.val.num_misreports = __C.train.num_misreports

__C.train.num_instances = __C.train.num_batches * __C.train.batch_size
__C.val.num_instances = __C.val.num_batches * __C.val.batch_size

__C.test.num_instances = __C.test.num_batches * __C.test.batch_size
if not osp.exists(__C.dir_name): os.mkdir(__C.dir_name)
