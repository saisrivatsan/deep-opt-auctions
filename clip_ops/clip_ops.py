from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

def clip_op_1215(x):
    clip_op_1 = tf.assign(x[:, :, 0, :], tf.clip_by_value(x[:, :, 0, :], 1.0, 2.0))
    clip_op_2 = tf.assign(x[:, :, 1, :], tf.clip_by_value(x[:, :, 1, :], 1.0, 5.0))
    clip_op = tf.group([clip_op_1, clip_op_2])
    return clip_op

def clip_op_01(x):
    clip_op = tf.assign(x, tf.clip_by_value(x, 0.0, 1.0))
    return clip_op
    
def clip_op_12(x):
    clip_op = tf.assign(x, tf.clip_by_value(x, 1.0, 2.0))
    return clip_op
    
def clip_op_23(x):
    clip_op = tf.assign(x, tf.clip_by_value(x, 2.0, 3.0))
    return clip_op