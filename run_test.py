from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from cfgs.config_3x10 import cfg
from nets.additive_net import Net
from data.uniform_01_generator import Generator
from trainer.trainer import Trainer


net = Net(cfg)
generator = Generator(cfg, 'test')
clip_op_lambda = (lambda x: tf.assign(x, tf.clip_by_value(x, 0.0, 1.0)))
m = Trainer(cfg, "test", net, clip_op_lambda)
m.test(generator)
