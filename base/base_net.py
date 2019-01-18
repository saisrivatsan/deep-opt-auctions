from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging
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


class BaseNet(object):
    
    def __init__(self, config):
        self.config = config
        
        """ Set initializer """
        if self.config.net.init is 'None': init  = None
        elif self.config.net.init == 'gu': init = tf.keras.initializers.glorot_uniform()
        elif self.config.net.init == 'gn': init = tf.keras.initializers.glorot_normal()
        elif self.config.net.init == 'hu': init = tf.keras.initializers.he_uniform()
        elif self.config.net.init == 'hn': init = tf.keras.initializers.he_normal()
        self.init = init
        
        if self.config.net.activation == 'tanh': activation = lambda *x: tf.tanh(*x)
        elif self.config.net.activation == 'relu': activation = lambda *x: tf.nn.relu(*x)
        self.activation = activation        
               
    def build_net(self):
        """
        Initializes network variables
        """
        raise NotImplementedError
        
    def inference(self, x):
        """ 
        Inference 
        """
        raise NotImplementedError
        
            
            
