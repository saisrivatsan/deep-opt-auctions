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

        num_a_layers = self.config.net.num_a_layers        
        num_p_layers = self.config.net.num_p_layers

        num_a_hidden_units = self.config.net.num_a_hidden_units
        num_p_hidden_units = self.config.net.num_p_hidden_units
        
                
        w_init = self.init
        b_init = tf.keras.initializers.Zeros()

        wd = None if "wd" not in self.config.train else self.config.train.wd
            
        # Alloc network weights and biases
        self.w_a = []
        self.b_a = []

        # Pay network weights and biases
        self.w_p = []
        self.b_p = []

        num_in = num_agents * num_items
   

        with tf.variable_scope("alloc"):
           
            # Input Layer
            self.w_a.append(create_var("w_a_0", [num_in, num_a_hidden_units], initializer = w_init, wd = wd))

            # Hidden Layers
            for i in range(1, num_a_layers - 1):
                wname = "w_a_" + str(i)
                self.w_a.append(create_var(wname, [num_a_hidden_units, num_a_hidden_units], initializer = w_init, wd = wd))
                
            # Output Layer
            wname = "w_a_" + str(num_a_layers - 1)   
            self.w_a.append(create_var(wname, [num_a_hidden_units, (num_agents + 1) * (num_items + 1)], initializer = w_init, wd = wd))

            # Biases
            for i in range(num_a_layers - 1):
                wname = "b_a_" + str(i)
                self.b_a.append(create_var(wname, [num_a_hidden_units], initializer = b_init))
                
            wname = "b_a_" + str(num_a_layers - 1)   
            self.b_a.append(create_var(wname, [(num_agents + 1) * (num_items + 1)], initializer = b_init))

            
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
 
        x_in = tf.reshape(x, [-1, self.config.num_agents * self.config.num_items])

      
        # Allocation Network
        a = tf.matmul(x_in, self.w_a[0]) + self.b_a[0]
        a = self.activation(a, 'alloc_act_0')
        activation_summary(a)
        
        for i in range(1, self.config.net.num_a_layers - 1):
            a = tf.matmul(a, self.w_a[i]) + self.b_a[i]
            a = self.activation(a, 'alloc_act_' + str(i))                    
            activation_summary(a)

        a = tf.matmul(a, self.w_a[-1]) + self.b_a[-1]
        a = tf.nn.softmax(tf.reshape(a, [-1, self.config.num_agents + 1, self.config.num_items + 1]), axis = 1)
        a = tf.slice(a, [0,0,0], size=[-1, self.config.num_agents, self.config.num_items], name = 'alloc_out')
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
        
        u = tf.reduce_sum(a * tf.reshape(x, [-1, self.config.num_agents, self.config.num_items]), axis = -1)
        p = p * u
        activation_summary(p)
        
        return a, p
