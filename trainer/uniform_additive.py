from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from data.uniform_01_generator import Generator
from nets.addtive_net import Net


class Trainer(BaseTrainer):
	def __init__(self, config):
        super(Trainer, self).__init__(config, mode)

    def init_generators(self):
        # Data generators
        if self.mode is "train":
            self.train_gen = Generator(self.config, 'train')
            self.val_gen = Generator(self.config, 'val')
        else:
            self.test_gen = Generator(self.config, 'test')

    def init_net(self):
        self.net = Net(config)

    def clip_op(self)
        return tf.assign(self.adv_var, tf.clip_by_value(adv_var, 0.0, 1.0))