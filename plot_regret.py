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

print("Restore-iter: %d"%(int(sys.argv[1])))
restore_iter = int(sys.argv[1])

cfg = additive_1x2_uniform_config.cfg
Net = additive_net.Net
Generator = uniform_01_generator.Generator
clip_op_lambda = (lambda x: clip_op_01(x))
Trainer = trainer.Trainer

cfg.test.restore_iter = restore_iter       
net = Net(cfg)
generator = Generator(cfg, 'test')
m = Trainer(cfg, "test", net, clip_op_lambda)
m.test(generator)
