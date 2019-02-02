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

        num_func = self.config.net.num_func
        num_max_units = self.config.net.num_max_units
        num_agents = self.config.num_agents
            
        wd = None if "wd" not in self.config.train else self.config.train.wd
               
        with tf.variable_scope("myersonNet"):
            self.w = create_var("w", [num_max_units, num_func, num_agents], initializer = self.w_init, wd = wd)
            self.b = create_var("b", [num_max_units, num_func, num_agents], initializer = self.b_init, wd = wd)

            
    def inference(self, x):
        """
        Inference
        """
        
        num_func = self.config.net.num_func
        num_max_units = self.config.net.num_max_units
        num_agents = self.config.num_agents
        
        batch_size = self.config[self.mode].batch_size
                
        W = tf.tile(self.w[tf.newaxis, ...], [batch_size, 1, 1, 1])
        B = tf.tile(self.b[tf.newaxis, ...], [batch_size, 1, 1, 1])        
        x = tf.tile(x[:, tf.newaxis, tf.newaxis, :], [1, num_max_units, num_func, 1])   
        
        vv = tf.reduce_min(tf.reduce_max(tf.multiply(x, tf.exp(W)) + B, axis = 2), axis = 1)
        
        a = tf.pad(vv, [[0,0],[0,1]], "CONSTANT")
        if self.mode is 'train':
            a = tf.nn.softmax(a * self.config.net.eps, axis = -1)
        if self.mode is 'test':
            a = tf.one_hot(tf.argmax(a, axis = -1), num_agents + 1)        
        a = tf.slice(a, [0, 0], [-1, num_agents])
        
                        
        wp = tf.matrix_diag(np.float32(np.ones((num_agents, num_agents)) - np.identity(num_agents)))
        y = tf.tile(vv[tf.newaxis, :, :], [num_agents, 1, 1]) 
        y = tf.matmul(y, wp)      
        y = tf.transpose(tf.reduce_max(y, axis = -1))
        y = tf.nn.relu(y)
        
        ## Decode the payment
        y = tf.tile(y[:, tf.newaxis, tf.newaxis, :], [1, num_max_units, num_func, 1])
        p = tf.reduce_max(tf.reduce_min(tf.multiply(y - B, tf.exp(-W)), axis = 2), axis = 1)
        p = tf.multiply(a, p)
                    
        return a, p, vv