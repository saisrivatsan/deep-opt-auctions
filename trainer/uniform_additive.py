from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from nets.additive_net import Net
from base.base_trainer import BaseTrainer
from data.uniform_01_generator import Generator

class Trainer(BaseTrainer):
    def __init__(self, config, mode):
        super(Trainer, self).__init__(config, mode)
        self.build_model()

    def init_generators(self):
        if self.mode is "train":
            self.train_gen = Generator(self.config, 'train')
            self.val_gen = Generator(self.config, 'val')
        else:
            self.test_gen = Generator(self.config, 'test')

    def init_net(self):
        self.net = Net(self.config)

    def get_clip_op(self):
        return tf.assign(self.adv_var, tf.clip_by_value(self.adv_var, 0.0, 1.0))