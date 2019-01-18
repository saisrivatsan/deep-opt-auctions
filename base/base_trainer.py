from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import logging
import numpy as np
import tensorflow as tf


class BaseTrainer(object):

    def __init__(self, config, mode):
        self.config = config
        self.mode = mode

        if self.mode == "train":
            log_suffix = '_' + str(self.config.train.restore_iter) if self.config.train.restore_iter > 0 else ''
            self.log_fname = os.path.join(self.config.dir_name, 'train' + log_suffix + '.txt')
        else:
            log_suffix = "_iter_" + str(self.config.test.restore_iter) + "_m_" + str(self.config.test.num_misreports) + "_gd_" + str(self.config.test.gd_iter)
            self.log_fname = os.path.join(self.config.dir_name, "test" + log_suffix + ".txt")   
        

    def init_logger(self):


        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        handler = logging.FileHandler(self.log_fname, 'w')
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        self.logger = logger

    def compute_rev(self, pay):
        """ Given payment (pay), computes revenue
            Input params:
                pay: [num_batches, num_agents]
            Output params:
                revenue: scalar
        """
        return tf.reduce_mean(tf.reduce_sum(pay, axis=-1))

    def compute_utility(self, x, alloc, pay):
        """ Given input valuation (x), payment (pay) and allocation (alloc), computes utility
            Input params:
                x: [num_batches, num_agents, num_items]
                a: [num_batches, num_agents, num_items]
                p: [num_batches, num_agents]
            Output params:
                utility: [num_batches, num_agents]
        """
        return tf.reduce_sum(tf.multiply(alloc, x), axis=-1) - pay


    def get_misreports(self, x, adv_var, adv_shape):

        num_misreports = adv_shape[1]
        adv = tf.tile(tf.expand_dims(adv_var, 0), [self.config.num_agents, 1, 1, 1, 1])
        x_mis = tf.tile(x, [self.config.num_agents * num_misreports, 1, 1])
        x_r = tf.reshape(x_mis, adv_shape)
        y = x_r * (1 - self.adv_mask) + adv * self.adv_mask
        misreports = tf.reshape(y, [-1, self.config.num_agents, self.config.num_items])
        return x_mis, misreports

    
    def init_generators(self):
        raise NotImplementedError

    def init_net(self):
        raise NotImplementedError

    def get_clip_op(self):
        raise NotImplementedError


    def init_graph(self):
       
        x_shape = [self.config[self.mode].batch_size, self.config.num_agents, self.config.num_items]
        adv_shape = [self.config.num_agents, self.config[self.mode].num_misreports, self.config[self.mode].batch_size, self.config.num_agents, self.config.num_items]
        adv_var_shape = [ self.config[self.mode].num_misreports, self.config[self.mode].batch_size, self.config.num_agents, self.config.num_items]
        u_shape = [self.config.num_agents, self.config[self.mode].num_misreports, self.config[self.mode].batch_size, self.config.num_agents]

        # Placeholders
        self.x = tf.placeholder(tf.float32, shape=x_shape, name='x')
        self.adv_init = tf.placeholder(tf.float32, shape=adv_var_shape, name='adv_init')
        
        self.adv_mask = np.zeros(adv_shape)
        self.adv_mask[np.arange(self.config.num_agents), :, :, np.arange(self.config.num_agents), :] = 1.0
        
        self.u_mask = np.zeros(u_shape)
        self.u_mask[np.arange(self.config.num_agents), :, :, np.arange(self.config.num_agents)] = 1.0
        
        with tf.variable_scope('adv_var'):
            self.adv_var = tf.get_variable('adv_var', shape = adv_var_shape, dtype = tf.float32)
            
            
        # Misreports
        x_mis, misreports = self.get_misreports(self.x, self.adv_var, adv_shape)
        
        # Get mechanism for true valuation: Allocation and Payment
        self.alloc, self.pay = self.net.forward(self.x)
        
        # Get mechanism for misreports: Allocation and Payment
        a_mis, p_mis = self.net.forward(misreports)
        
        # Utility
        utility = self.compute_utility(self.x, self.alloc, self.pay)
        utility_mis = self.compute_utility(x_mis, a_mis, p_mis)
        
        # Regret Computation
        u_mis = tf.reshape(utility_mis, u_shape) * self.u_mask
        utility_true = tf.tile(utility, [self.config.num_agents * self.config[self.mode].num_misreports, 1])
        excess_from_utility = tf.nn.relu(tf.reshape(utility_mis - utility_true, u_shape) * self.u_mask)
        rgt = tf.reduce_mean(tf.reduce_max(excess_from_utility, axis=(1, 3)), axis=1)
    
        #Metrics
        revenue = self.compute_rev(self.pay)
        rgt_mean = tf.reduce_mean(rgt)
        irp_mean = tf.reduce_mean(tf.nn.relu(-utility))

        # Variable Lists
        alloc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='alloc')
        pay_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='pay')
        var_list = alloc_vars + pay_vars


        
        if self.mode is "train":

            w_rgt_init_val = 0.0 if "w_rgt_init_val" not in self.config.train else self.config.train.w_rgt_init_val

            with tf.variable_scope('lag_var'):
                self.w_rgt = tf.Variable(np.ones(self.config.num_agents).astype(np.float32) * w_rgt_init_val, 'w_rgt')


            update_rate = tf.Variable(self.config.train.update_rate, trainable = False)
            self.up_op = update_rate.assign(update_rate + self.config.train.up_op_add)

      
            # Loss Functions
            rgt_penalty = update_rate * tf.reduce_sum(tf.square(rgt)) / 2.0        
            lag_loss = tf.reduce_sum(self.w_rgt * rgt)
        

            loss_1 = -revenue + rgt_penalty + lag_loss
            loss_2 = -tf.reduce_sum(u_mis)
            loss_3 = -lag_loss

            reg_losses = tf.get_collection('reg_losses')
            if len(reg_losses) > 0:
                reg_loss_mean = tf.reduce_mean(reg_losses)
                loss_1 = loss_1 + reg_loss_mean

             
            learning_rate = tf.Variable(self.config.train.learning_rate, trainable = False)
        
            # Optimizer
            opt_1 = tf.train.AdamOptimizer(learning_rate)
            opt_2 = tf.train.AdamOptimizer(self.config.train.gd_lr)
            opt_3 = tf.train.GradientDescentOptimizer(update_rate)


            # Train ops
            self.train_step_1  = opt_1.minimize(loss_1, var_list = var_list)
            self.train_gd_step = opt_2.minimize(loss_2, var_list = [self.adv_var])
            self.lag_update    = opt_3.minimize(loss_3, var_list = [self.w_rgt])
            
            # Val ops
            val_opt = tf.train.AdamOptimizer(self.config.val.gd_lr)
            self.val_gd_step = val_opt.minimize(loss_2, var_list = [self.adv_var])       

            # Reset ops
            self.train_reset_opt = tf.variables_initializer(opt_2.variables()) 
            self.val_reset_opt = tf.variables_initializer(val_opt.variables())

            # Metrics
            self.metrics = [revenue, rgt_mean, rgt_penalty, lag_loss, loss_1, tf.reduce_mean(self.w_rgt), update_rate]
            self.metric_names = ["Revenue", "Regret", "Reg_Loss", "Lag_Loss", "Net_Loss", "w_rgt_mean", "update_rate"]
            
            #Summary
            tf.summary.scalar('revenue', revenue)
            tf.summary.scalar('regret', rgt_mean)
            tf.summary.scalar('reg_loss', rgt_penalty)
            tf.summary.scalar('lag_loss', lag_loss)
            tf.summary.scalar('net_loss', loss_1)
            tf.summary.scalar('w_rgt_mean', tf.reduce_mean(self.w_rgt))
            if len(reg_losses) > 0: tf.summary.scalar('reg_loss', reg_loss_mean)

            self.merged = tf.summary.merge_all()
            self.saver = tf.train.Saver(max_to_keep = self.config.train.max_to_keep)
        
        elif self.mode is "test":

            loss = -tf.reduce_sum(u_mis)
            opt = tf.train.AdamOptimizer(self.config.test.gd_lr)
            self.test_gd_step = opt.minimize(loss, var_list = [self.adv_var])
            self.test_reset_opt = tf.variables_initializer(opt.variables())

            # Metrics
            self.metrics = [revenue, rgt_mean, irp_mean]
            self.metric_names = ["Revenue", "Regret", "IRP"]
            self.saver = tf.train.Saver(var_list = var_list)
            

        # Helper ops post GD steps
        self.assign_op = tf.assign(self.adv_var, self.adv_init)
        self.clip_op = self.get_clip_op()
        
    def build_model(self):

        # Set Seeds for reproducibility
        np.random.seed(self.config[self.mode].seed)
        tf.set_random_seed(self.config[self.mode].seed)
        
        # Init Logger
        self.init_logger()

        # Init Net
        self.init_net()

        # Init TF-graph
        self.init_graph()

        # Init generators
        self.init_generators()   


    def train(self):

        iter = self.config.train.restore_iter
        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        train_writer = tf.summary.FileWriter(self.config.dir_name, sess.graph)
        
        if iter > 0:
            model_path = os.path.join(self.config.dir_name, 'model-' + str(iter))
            self.saver.restore(sess, model_path)

        if iter == 0:
            self.train_gen.save_data(0)

        time_elapsed = 0.0
        while iter < (self.config.train.max_iter):
             
            # Get a mini-batch
            X, ADV, perm = next(self.train_gen.gen_func)
                
            if iter == 0: sess.run(self.lag_update, feed_dict = {self.x : X})
 

            tic = time.time()    
            
            # Get Best Mis-report
            sess.run(self.assign_op, feed_dict = {self.adv_init: ADV})                                        
            for _ in range(self.config.train.gd_iter):
                sess.run(self.train_gd_step, feed_dict = {self.x: X})
                sess.run(self.clip_op)
            sess.run(self.train_reset_opt)

            if self.config.train.data is "fixed" and self.config.train.adv_reuse:
                self.train_gen.update_adv(perm, sess.run(self.adv_var))

            # Update network params
            sess.run(self.train_step_1, feed_dict = {self.x: X})
                
            if iter==0:
                summary = sess.run(self.merged, feed_dict = {self.x: X})
                train_writer.add_summary(summary, iter) 

            iter += 1

            # Run Lagrange Update
            if iter % self.config.train.update_frequency == 0:
                sess.run(self.lag_update, feed_dict = {self.x:X})
                

            if iter % self.config.train.up_op_frequency == 0:
                sess.run(self.up_op)

            toc = time.time()
            time_elapsed += (toc - tic)
                        
            if ((iter % self.config.train.save_iter) == 0) or (iter == self.config.train.max_iter): 
                self.saver.save(sess, os.path.join(self.config.dir_name,'model'), global_step = iter) 
                self.train_gen.save_data(iter)

            if (iter % self.config.train.print_iter) == 0:
                # Train Set Stats
                summary = sess.run(self.merged, feed_dict = {self.x: X})
                train_writer.add_summary(summary, iter)
                metric_vals = sess.run(self.metrics, feed_dict = {self.x: X})
                fmt_vals = tuple([ item for tup in zip(self.metric_names, metric_vals) for item in tup ])
                log_str = "TRAIN-BATCH Iter: %d, t = %.4f"%(iter, time_elapsed) + ", %s: %.6f"*len(self.metric_names)%fmt_vals
                self.logger.info(log_str)

            if (iter % self.config.val.print_iter) == 0:
                #Validation Set Stats
                metric_tot = np.zeros(len(self.metric_names))         
                for _ in range(self.config.val.num_batches):
                    X, ADV, _ = next(self.val_gen.gen_func) 
                    sess.run(self.assign_op, feed_dict = {self.adv_init: ADV})               
                    for k in range(self.config.val.gd_iter):
                        sess.run(self.val_gd_step, feed_dict = {self.x: X})
                        sess.run(self.clip_op)
                    sess.run(self.val_reset_opt)                                   
                    metric_vals = sess.run(self.metrics, feed_dict = {self.x: X})
                    metric_tot += metric_vals
                    
                metric_tot = metric_tot/self.config.val.num_batches
                fmt_vals = tuple([ item for tup in zip(self.metric_names, metric_tot) for item in tup ])
                log_str = "VAL-%d"%(iter) + ", %s: %.6f"*len(self.metric_names)%fmt_vals
                self.logger.info(log_str)

    def test(self, X_tst = None, ADV_tst = None):

        iter = self.config.test.restore_iter
        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        model_path = os.path.join(self.config.dir_name,'model-' + str(iter))
        self.saver.restore(sess, model_path)

        #Test-set Stats
        time_elapsed = 0
            
        metric_tot = np.zeros(len(self.metric_names))

        if X_tst is not None:
            alloc_tst = np.zeros(X_tst.shape)
            pay_tst = np.zeros(X_tst.shape[:-1])
           
        tmp = [] #chksum           

        for i in range(self.config.test.num_batches):
            tic = time.time()
            X, ADV, perm = next(self.test_gen.gen_func)
            sess.run(self.assign_op, feed_dict = {self.adv_init: ADV})
                    
            for k in range(self.config.test.gd_iter):
                sess.run(self.test_gd_step, feed_dict = {self.x: X})
                sess.run(self.clip_op)

            sess.run(self.test_reset_opt)        
                
            metric_vals = sess.run(self.metrics, feed_dict = {self.x: X})
            
            if X_tst is not None:
                A, P = sess.run([self.alloc, self.pay], feed_dict = {self.x:X})
                alloc_tst[perm, :, :] = A
                pay_tst[perm, :] = P
                    
            metric_tot += metric_vals
            toc = time.time()
            time_elapsed += (toc - tic)

            fmt_vals = tuple([ item for tup in zip(self.metric_names, metric_vals) for item in tup ])
            log_str = "TEST BATCH-%d: t = %.4f"%(i, time_elapsed) + ", %s: %.6f"*len(self.metric_names)%fmt_vals
            self.logger.info(log_str)
        
        metric_tot = metric_tot/self.config.test.num_batches
        fmt_vals = tuple([ item for tup in zip(self.metric_names, metric_tot) for item in tup ])
        log_str = "TEST ALL-%d: t = %.4f"%(iter, time_elapsed) + ", %s: %.6f"*len(self.metric_names)%fmt_vals
        self.logger.info(log_str)
            
        if X_tst is not None:
            np.save(os.path.join(self.config.dir_name, 'alloc_tst_' + str(iter)), alloc_tst)
            np.save(os.path.join(self.config.dir_name, 'pay_tst_' + str(iter)), pay_tst)
            
        
        
        
