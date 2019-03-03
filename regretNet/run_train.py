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

print("Setting: %s"%(sys.argv[1]))
setting = sys.argv[1]

if setting == "additive_1x2_uniform":
    cfg = additive_1x2_uniform_config.cfg
    Net = additive_net.Net
    Generator = uniform_01_generator.Generator
    clip_op_lambda = (lambda x: clip_op_01(x))
    Trainer = trainer.Trainer

elif setting == "unit_1x2_uniform_23":
    cfg = unit_1x2_uniform_23_config.cfg
    Net = unit_net.Net
    Generator = uniform_23_generator.Generator
    clip_op_lambda = (lambda x: clip_op_23(x))
    Trainer = trainer.Trainer

elif setting == "additive_2x2_uniform":
    cfg = additive_2x2_uniform_config.cfg
    Net = additive_net.Net
    Generator = uniform_01_generator.Generator
    clip_op_lambda = (lambda x: clip_op_01(x))
    Trainer = trainer.Trainer

elif setting == "additive_2x3_uniform":
    cfg = additive_2x3_uniform_config.cfg
    Net = additive_net.Net
    Generator = uniform_01_generator.Generator
    clip_op_lambda = (lambda x: clip_op_01(x))
    Trainer = trainer.Trainer

elif setting == "additive_3x10_uniform":
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
    clip_op_lambda = (lambda x: clip_op_12_15(x))
    Trainer = ca12_2x2.Trainer

elif setting == "CA_sym_uniform_12":
    cfg = CA_sym_uniform_12_config.cfg
    Net = ca2x2_net.Net
    Generator = CA_sym_uniform_12_generator.Generator
    clip_op_lambda = (lambda x: clip_op_12(x))
    Trainer = ca12_2x2.Trainer

elif setting == "additive_1x2_uniform_416_47":
    cfg = additive_1x2_uniform_416_47_config.cfg
    Net = additive_net.Net
    Generator = uniform_416_47_generator.Generator
    clip_op_lambda = (lambda x: clip_op_416_47(x))
    Trainer = trainer.Trainer
    
elif setting == "additive_1x2_uniform_triangle":
    cfg = additive_1x2_uniform_triangle_config.cfg
    Net = additive_net.Net
    Generator = uniform_triangle_01_generator.Generator
    clip_op_lambda = (lambda x: clip_op_triangle_01(x))
    Trainer = trainer.Trainer
    
elif setting == "unit_1x2_uniform":
    cfg = unit_1x2_uniform_config.cfg
    Net = unit_net.Net
    Generator = uniform_01_generator.Generator
    clip_op_lambda = (lambda x: clip_op_01(x))
    Trainer = trainer.Trainer

elif setting == "additive_1x10_uniform":
    cfg = additive_1x10_uniform_config.cfg
    Net = additive_net.Net
    Generator = uniform_01_generator.Generator
    clip_op_lambda = (lambda x: clip_op_01(x))
    Trainer = trainer.Trainer

elif setting == "additive_1x2_uniform_04_03":
    cfg = additive_1x2_uniform_04_03_config.cfg
    Net = additive_net.Net
    Generator = uniform_04_03_generator.Generator
    clip_op_lambda = (lambda x: clip_op_04_03(x))
    Trainer = trainer.Trainer

elif setting == "unit_2x2_uniform":
    cfg = unit_2x2_uniform_config.cfg
    Net = unit_net.Net
    Generator = uniform_01_generator.Generator
    clip_op_lambda = (lambda x: clip_op_01(x))
    Trainer = trainer.Trainer
    
else:
    print("None selected")
    sys.exit(0)
    

net = Net(cfg)
generator = [Generator(cfg, 'train'), Generator(cfg, 'val')]
m = Trainer(cfg, "train", net, clip_op_lambda)
m.train(generator)
