from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def clip_op_01(x):
    clip_op = tf.assign(x, tf.clip_by_value(x, 0.0, 1.0))
    return clip_op
    
def clip_op_12(x):
    clip_op = tf.assign(x, tf.clip_by_value(x, 1.0, 2.0))
    return clip_op
    
def clip_op_23(x):
    clip_op = tf.assign(x, tf.clip_by_value(x, 2.0, 3.0))
    return clip_op

def clip_op_12_15(x):
    clip_op_1 = tf.assign(x[:, :, 0, :], tf.clip_by_value(x[:, :, 0, :], 1.0, 2.0))
    clip_op_2 = tf.assign(x[:, :, 1, :], tf.clip_by_value(x[:, :, 1, :], 1.0, 5.0))
    clip_op = tf.group([clip_op_1, clip_op_2])
    return clip_op

def clip_op_416_47(x):
    clip_op_1 = tf.assign(x[:, :, :, 0], tf.clip_by_value(x[:, :, :, 0], 4.0, 16.0))
    clip_op_2 = tf.assign(x[:, :, :, 1], tf.clip_by_value(x[:, :, :, 1], 4.0, 7.0))
    clip_op = tf.group([clip_op_1, clip_op_2])
    return clip_op

def clip_op_04_03(x):
    clip_op_1 = tf.assign(x[:, :, :, 0], tf.clip_by_value(x[:, :, :, 0], 0.0, 4.0))
    clip_op_2 = tf.assign(x[:, :, :, 1], tf.clip_by_value(x[:, :, :, 1], 0.0, 3.0))
    clip_op = tf.group([clip_op_1, clip_op_2])
    return clip_op

def clip_op_triangle_01_numpy(x):
    x_shape = x.shape
    x = np.reshape(x, [-1, 2])
    
    invalid_idx = np.where( (x[:,0]<0) | (x[:,1]<0) | (x.sum(-1)>=1) )

    x_invalid = x[invalid_idx]

    p = np.zeros((x_invalid.shape[0], 3, 2))
    d = np.zeros((x_invalid.shape[0], 3))
    t = np.zeros((x_invalid.shape[0], 3))
    t[:, 0] = (x_invalid[:, 0] - x_invalid[:, 1] + 1.0)/2.0
    t[:, 1] = (1 - x_invalid[:, 1])
    t[:, 2] = (1 - x_invalid[:, 0])
    t = np.clip(t, 0.0, 1.0)

    A = np.array([[0,1]]).T
    B = np.array([[1,0]]).T
    O = np.array([[0,0]]).T
    pts_x = [A, A, B]
    pts_y = [B, O, O]

    for i in range(3):
        p[:, i, :] = ((1 - t[:, i]) * pts_x[i] +  t[:, i] * pts_y[i]).T
        d[:, i] = np.sum((x_invalid - p[:, i, :])**2, -1)

    sel_p = p[np.arange(x_invalid.shape[0]), np.argmin(d, -1), :]

    x[invalid_idx] = sel_p
    x = np.reshape(x, x_shape)
    return x


def clip_op_triangle_01(x):
    y = tf.py_func(clip_op_triangle_01_numpy, [x], tf.float32)
    clip_op = tf.assign(x, y)
    return clip_op
    
