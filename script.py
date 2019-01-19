from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import tensorflow as tf


from nets import *
from cfgs import *
from data import *
from clip_ops.clip_ops import *
from trainer import *

setting = "additive_5x10_uniform"

if setting == "additive_3x10_uniform":
    cfg = additive_3x10_uniform_config.cfg
    Net = additive_net.Net
    Generator = uniform_01_generator.Generator
    clip_op_lambda = (lambda x: clip_op_01(x))
    Trainer = trainer.Trainer
    
elif setting == "additive_5x10_uniform":
    cfg = additive_5x10_uniform_config.cfg
    Net = additive_net.Net
    Generator = uniform_01_generator.Generator
    clip_op_lambda = (lambda x: clip_op_01(x))
    Trainer = trainer.Trainer

elif setting == "CA_asym_uniform_12_15":
    cfg = CA_asym_uniform_12_15_config.cfg
    Net = ca2x2_net.Net
    Generator = CA_asym_uniform_12_15_generator.Generator
    clip_op_lambda = (lambda x: clip_op_1215(x))
    Trainer = ca12_2x2.Trainer

elif setting == "CA_sym_uniform_12":
    cfg = CA_sym_uniform_12_config.cfg
    Net = ca2x2_net.Net
    Generator = CA_sym_uniform_12_generator.Generator
    clip_op_lambda = (lambda x: clip_op_12(x))
    Trainer = ca12_2x2.Trainer
    
net = Net(cfg)
generator = [Generator(cfg, 'train'), Generator(cfg, 'val')]
m = Trainer(cfg, "train", net, clip_op_lambda)
m.train(generator)
