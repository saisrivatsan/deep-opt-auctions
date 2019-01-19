from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from cfgs.config_ca12 import cfg
from nets.ca2x2_net import Net
from data.ca_asym_uniform_12_15_generator import Generator
from trainer.ca12_2x2 import Trainer

def clip_op_(x):
    clip_op_1 = tf.assign(x[:, :, 0, :], tf.clip_by_value(x[:, :, 0, :], 1.0, 2.0))
    clip_op_2 = tf.assign(x[:, :, 1, :], tf.clip_by_value(x[:, :, 1, :], 1.0, 5.0))
    clip_op = tf.group([clip_op_1, clip_op_2])
    return clip_op

clip_op_lambda = (lambda x: clip_op_(x))

net = Net(cfg)
generator = Generator(cfg, 'test')

m = Trainer(cfg, "test", net, clip_op_lambda)
m.test(generator)
