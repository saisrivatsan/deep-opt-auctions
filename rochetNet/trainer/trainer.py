from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import logging
import numpy as np
import tensorflow as tf


class Trainer(object):

    def __init__(self, config, mode, net):
        self.config = config
        self.mode = mode
        
        # Create output-dir
        if not os.path.exists(self.config.dir_name): os.mkdir(self.config.dir_name)

        if self.mode == "train":
            log_suffix = '_' + str(self.config.train.restore_iter) if self.config.train.restore_iter > 0 else ''
            self.log_fname = os.path.join(self.config.dir_name, 'train' + log_suffix + '.txt')
        else:
            log_suffix = "_iter_" + str(self.config.test.restore_iter)
            self.log_fname = os.path.join(self.config.dir_name, "test" + log_suffix + ".txt")
            
        # Set Seeds for reproducibility
        np.random.seed(self.config[self.mode].seed)
        tf.set_random_seed(self.config[self.mode].seed)
        
        # Init Logger
        self.init_logger()

        # Init Net
        self.net = net
        
        # Init TF-graph
        self.init_graph()
              
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
        return tf.reduce_mean(pay)
    
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


    def init_graph(self):
       
        x_shape = [self.config[self.mode].batch_size, self.config.num_items]
        
        # Placeholders
        self.x = tf.placeholder(tf.float32, shape=x_shape, name='x')
        
        # Get mechanism for true valuation: Allocation and Payment
        self.alloc, self.pay = self.net.inference(self.x)
        
        #Metrics
        revenue = self.compute_rev(self.pay)
        utility = self.compute_utility(self.x, self.alloc, self.pay) 
        irp_mean = tf.reduce_mean(tf.nn.relu(-utility))

        loss = -revenue
        reg_losses = tf.get_collection('reg_losses')
        if len(reg_losses) > 0:
            reg_loss_mean = tf.reduce_mean(reg_losses)
            loss += reg_loss_mean

        learning_rate = tf.Variable(self.config.train.learning_rate, trainable = False)

        # Optimizer
        opt = tf.train.AdamOptimizer(learning_rate)

        # Train ops
        self.train_op  = opt.minimize(loss)

        # Metrics
        self.metrics = [loss, revenue]
        self.metric_names = ["Net_Loss", "Revenue"]

        #Summary
        tf.summary.scalar('revenue', revenue)
        tf.summary.scalar('net_loss', loss)
        if len(reg_losses) > 0: tf.summary.scalar('reg_loss', reg_loss_mean)

        self.merged = tf.summary.merge_all()
        self.saver = tf.train.Saver(max_to_keep = self.config.train.max_to_keep)
        
        self.clip_op = tf.assign(self.net.alpha, tf.clip_by_value(self.net.alpha, 0.0, 1.0))
        
    def train(self, generator):
        """
        Runs training
        """
        
        self.train_gen, self.val_gen = generator
        
        iter = self.config.train.restore_iter
        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        train_writer = tf.summary.FileWriter(self.config.dir_name, sess.graph)
        
        if iter > 0:
            model_path = os.path.join(self.config.dir_name, 'model-' + str(iter))
            self.saver.restore(sess, model_path)

        if iter == 0:
            self.train_gen.save_data()
            self.saver.save(sess, os.path.join(self.config.dir_name,'model'), global_step = iter)

        time_elapsed = 0.0       
        while iter < (self.config.train.max_iter):
             
            tic = time.time()
                
            # Get a mini-batch
            X = next(self.train_gen.gen_func)
            sess.run(self.train_op, feed_dict = {self.x: X})
            #sess.run(self.clip_op)
            
            if iter==0:
                summary = sess.run(self.merged, feed_dict = {self.x: X})
                train_writer.add_summary(summary, iter)
                
                
            iter += 1

            toc = time.time()
            time_elapsed += (toc - tic)
                        
            if ((iter % self.config.train.save_iter) == 0) or (iter == self.config.train.max_iter): 
                self.saver.save(sess, os.path.join(self.config.dir_name,'model'), global_step = iter)

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
                    X = next(self.val_gen.gen_func)                                   
                    metric_vals = sess.run(self.metrics, feed_dict = {self.x: X})
                    metric_tot += metric_vals
                    
                metric_tot = metric_tot/self.config.val.num_batches
                fmt_vals = tuple([ item for tup in zip(self.metric_names, metric_tot) for item in tup ])
                log_str = "VAL-%d"%(iter) + ", %s: %.6f"*len(self.metric_names)%fmt_vals
                self.logger.info(log_str)

    def test(self, generator):
        """
        Runs test
        """
        
        # Init generators
        self.test_gen = generator

        iter = self.config.test.restore_iter
        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        model_path = os.path.join(self.config.dir_name,'model-' + str(iter))
        self.saver.restore(sess, model_path)

        #Test-set Stats
        time_elapsed = 0          
        metric_tot = np.zeros(len(self.metric_names))

        if self.config.test.save_output:
            alloc_tst = np.zeros(self.test_gen.X.shape)
            pay_tst = np.zeros(self.test_gen.X.shape[:-1])
                    
        for i in range(self.config.test.num_batches):
            tic = time.time()
            X = next(self.test_gen.gen_func)
            metric_vals = sess.run(self.metrics, feed_dict = {self.x: X})          
            if self.config.test.save_output:
                A, P = sess.run([self.alloc, self.pay], feed_dict = {self.x:X})
                perm =range(i * A.shape[0], (i + 1) * A.shape[0])
                alloc_tst[perm, :] = A
                pay_tst[perm] = P
                    
            metric_tot += metric_vals
            toc = time.time()
            time_elapsed += (toc - tic)

            fmt_vals = tuple([ item for tup in zip(self.metric_names, metric_vals) for item in tup ])
            #log_str = "TEST BATCH-%d: t = %.4f"%(i, time_elapsed) + ", %s: %.6f"*len(self.metric_names)%fmt_vals
            #self.logger.info(log_str)
        
        metric_tot = metric_tot/self.config.test.num_batches
        fmt_vals = tuple([ item for tup in zip(self.metric_names, metric_tot) for item in tup ])
        log_str = "TEST ALL-%d: t = %.4f"%(iter, time_elapsed) + ", %s: %.6f"*len(self.metric_names)%fmt_vals
        self.logger.info(log_str)
        if self.config.test.save_output:
            np.save(os.path.join(self.config.dir_name, 'alloc_tst_' + str(iter)), alloc_tst)
            np.save(os.path.join(self.config.dir_name, 'pay_tst_' + str(iter)), pay_tst)
            
        
        
        
