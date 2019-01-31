from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

def create_var(name, shape, dtype = tf.float32, initializer = None, wd = None, summaries = False, trainable = True):
    """ 
    Helper to create a Variable and summary if required
    Args:
        name: name of the variable
        shape: list of ints
        initializer: initializer for Variable
        wd: weight decay (adds regularizer)
        summaries: attach summaries
    Returns:
        Variable Tensor
    """
    
    var = tf.get_variable(name, shape = shape, dtype = dtype, initializer = initializer, trainable = trainable)
    
    """ Regularization """
    if wd is not None:
        reg = tf.multiply(tf.nn.l2_loss(var), wd, name = "{}/wd".format(var.op.name))
        tf.add_to_collection('reg_losses', reg)
   
    """ Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    if summaries:
        with tf.name_scope(name + '_summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    return var


def activation_summary(x):
    """ 
    Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.
    Args:
        x: Tensor
    """
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


class Net:

    def __init__(self, config, mode):
        self.config = config
        self.mode = mode
        self.w_init = tf.keras.initializers.RandomNormal()
        self.b_init = tf.keras.initializers.RandomUniform(*self.config.net.b_init)
        self.build_net()

    def build_net(self):
        """
        Initializes network variables
        """

        num_items = self.config.num_items
        num_hidden_units = self.config.net.num_hidden_units       
        wd = None if "wd" not in self.config.train else self.config.train.wd
        
        
        with tf.variable_scope("utility"):
            self.alpha = create_var("alpha", [num_items, num_hidden_units], initializer = self.w_init, wd = wd)            
            self.bias = create_var("bias", [num_hidden_units], initializer = self.b_init)


    def inference(self, x):
        """
        Inference
        """
        padding_w = tf.constant([[0, 0], [0, 1]])
        padding_b = tf.constant([[0, 1]])

        w = tf.pad(tf.nn.sigmoid(self.alpha), padding_w, "CONSTANT")
        b = tf.pad(self.bias, padding_b, "CONSTANT")

        utility = tf.matmul(x, w) + b
        U = tf.nn.softmax(utility * self.config.net.eps, -1)
        if self.mode is "train":
            a = tf.matmul(U, tf.transpose(w))
        else:
            a = tf.matmul(tf.one_hot(tf.argmax(utility, -1), self.config.net.num_hidden_units + 1), tf.transpose(w))
        p = tf.reduce_sum(tf.multiply(a, x), -1) - tf.reduce_max(utility, -1)
        
        return a, p