from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from base.base_net import *

class Net(BaseNet):

    def __init__(self, config):
        super(Net, self).__init__(config)
        self.build_net()

    def build_net(self):
        """
        Initializes network variables
        """

        num_agents = self.config.num_agents
        num_items = self.config.num_items

        num_a_hidden_units = self.config.net.num_a_hidden_units        
        num_p_hidden_units = self.config.net.num_p_hidden_units

        num_p_layers = self.config.net.num_p_layers
        num_a_layers = self.config.net.num_a_layers

        assert(num_agents == 2), "Only supports num_agents = 2"
        assert(num_items == 2), "Only supports num_items = 2"
        
        w_init = self.init
        b_init = tf.keras.initializers.Zeros()

        wd = None if "wd" not in self.config.train else self.config.train.wd
            
        # Alloc network weights and biases
        self.w_a = []
        self.b_a = []

        # Pay network weights and biases
        self.w_p = []
        self.b_p = []

        num_in = 6
       
        with tf.variable_scope("alloc"):

            # Input Layer
            self.w_a.append(create_var("w_a_0", [num_in, num_a_hidden_units], initializer = w_init, wd = wd))

            # Hidden Layers
            for i in range(1, num_a_layers - 1):
                wname = "w_a_" + str(i)
                self.w_a.append(create_var(wname, [num_a_hidden_units, num_a_hidden_units], initializer = w_init, wd = wd))
     
            # Last Layer alloc weights
            wname = "wi1_a_" + str(num_a_layers - 1)   
            self.wi1_a = create_var(wname, [num_a_hidden_units, 5], initializer = w_init, wd = wd)

            wname = "wi2_a_" + str(num_a_layers - 1)
            self.wi2_a = create_var(wname, [num_a_hidden_units, 5], initializer = w_init, wd = wd)

            wname = "wa1_a_" + str(num_a_layers - 1)
            self.wa1_a = create_var(wname, [num_a_hidden_units, 3], initializer = w_init, wd = wd)

            wname = "wa2_a_" + str(num_a_layers - 1)
            self.wa2_a = create_var(wname, [num_a_hidden_units, 3], initializer = w_init, wd = wd)

            # Biases
            for i in range(num_a_layers - 1):
                wname = "b_a_" + str(i)
                self.b_a.append(create_var(wname, [num_a_hidden_units], initializer = b_init))
                
            # Last Layer alloc bias

            wname = "bi1_a_" + str(num_a_layers - 1)
            self.bi1_a = create_var(wname, [5], initializer = b_init)

            wname = "bi2_a_" + str(num_a_layers - 1)
            self.bi2_a = create_var(wname, [5], initializer = b_init)

            wname = "ba1_a_" + str(num_a_layers - 1)   
            self.ba1_a = create_var(wname, [3], initializer = b_init)

            wname = "ba2_a_" + str(num_a_layers - 1)
            self.ba2_a = create_var(wname, [3], initializer = b_init)

            

        with tf.variable_scope("pay"):
            
            # Input Layer
            self.w_p.append(create_var("w_p_0", [num_in, num_p_hidden_units], initializer = w_init, wd = wd))

            # Hidden Layers
            for i in range(1, num_p_layers - 1):
                wname = "w_p_" + str(i)
                self.w_p.append(create_var(wname, [num_p_hidden_units, num_p_hidden_units], initializer = w_init, wd = wd))
                
            # Output Layer
            wname = "w_p_" + str(num_p_layers - 1)   
            self.w_p.append(create_var(wname, [num_p_hidden_units, num_agents], initializer = w_init, wd = wd))

            # Biases
            for i in range(num_p_layers - 1):
                wname = "b_p_" + str(i)
                self.b_p.append(create_var(wname, [num_p_hidden_units], initializer = b_init))
                
            wname = "b_p_" + str(num_p_layers - 1)   
            self.b_p.append(create_var(wname, [num_agents], initializer = b_init))
            
    def inference(self, x):
        """
        Inference
        """
        
        x_in = tf.reshape(x, [-1, 6])
        
        # Allocation Network 
        a = tf.matmul(x_in, self.w_a[0]) + self.b_a[0]
        a = self.activation(a, 'alloc_act_0')
        activation_summary(a)
        
        for i in range(1, self.config.net.num_a_layers - 1):
            a = tf.matmul(a, self.w_a[i]) + self.b_a[i]
            a = self.activation(a, 'alloc_act_' + str(i))                    
            activation_summary(a)

        # From Zhe's code
        a_item1_ = tf.nn.softmax(tf.matmul(a, self.wi1_a) + self.bi1_a)
        a_item2_ = tf.nn.softmax(tf.matmul(a, self.wi2_a) + self.bi2_a)
        
        a_agent1_bundle = tf.nn.softmax(tf.matmul(a, self.wa1_a) + self.ba1_a)
        a_agent2_bundle = tf.nn.softmax(tf.matmul(a, self.wa2_a) + self.ba2_a)
        
        a_agent1_ = tf.concat([tf.slice(a_item1_, [0,0], [-1,1]), tf.slice(a_item2_, [0,0], [-1,1]), tf.minimum(tf.slice(a_item1_, [0,2], [-1,1]), tf.slice(a_item2_, [0,2], [-1,1]))], axis = 1)
        a_agent2_ = tf.concat([tf.slice(a_item1_, [0,1], [-1,1]), tf.slice(a_item2_, [0,1], [-1,1]), tf.minimum(tf.slice(a_item1_, [0,3], [-1,1]), tf.slice(a_item2_, [0,3], [-1,1]))], axis = 1)
                
        a_agent1 = tf.minimum(a_agent1_, a_agent1_bundle)
        a_agent2 = tf.minimum(a_agent2_, a_agent2_bundle)
        
        a = tf.reshape(tf.concat([a_agent1, a_agent2], axis = 1), [-1, 2, 3])
        # Zhe's code End

        activation_summary(a)
        

        # Payment Network
        p = tf.matmul(x_in, self.w_p[0]) + self.b_p[0]
        p = self.activation(p, 'pay_act_0')                  
        activation_summary(p)

        for i in range(1, self.config.net.num_p_layers - 1):
            p = tf.matmul(p, self.w_p[i]) + self.b_p[i]
            p = self.activation(p, 'pay_act_' + str(i))                  
            activation_summary(p)

        p = tf.matmul(p, self.w_p[-1]) + self.b_p[-1]
        p = tf.sigmoid(p, 'pay_sigmoid')
        activation_summary(p)
        
        u = tf.reduce_sum(a * tf.reshape(x, [-1, 2, 3]), [-1])
        p = p * u
        activation_summary(p)
        
        return a, p
        
            
            
