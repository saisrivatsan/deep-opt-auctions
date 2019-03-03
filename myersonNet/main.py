from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# Import python packages
import itertools
import matplotlib
import time
matplotlib.use('Agg')
import numpy as np # operations on arrays and matrices
import pylab as pl # plots
import sys
import os

from math import sqrt
from nets.net import MyersonNet
from utils.plot import PlotOneItem
from utils.cfg import config_cl
from data.generatedata import Generator
################################################################################
# main
################################################################################


def main(args, dir_name):
    generate_train = Generator(args, seed_val = 1)
    generate_test = Generator(args, seed_val = 2)
    train_data = generate_train.generate_sample('train')
    test_data = generate_test.generate_sample('test')
    
    if args.num_item != 1:
        sys.exit("Input size wrong: Number of item is larger than 1,\
            MyersonNet only holds for single-item auction.")       
    
    auction_nn = MyersonNet(args, train_data, test_data)
        
    # t = time.time()
    mechanism = auction_nn.nn_train()
    # print('Running time:' + str(time.time() - t))
    test_perform = auction_nn.nn_test(test_data, mechanism)
    
    # np.save(dir_name + '/results', (mechanism, train_perform, test_perform))
    # np.save(dir_name + '/data', (train_data, test_data))
    
    plot_figure(args, dir_name, test_data, test_perform)
        
    
################################################################################
# Plot results and allocations
################################################################################
def plot_figure(args, dir_name, data, perform):
    # _, _, test_perform = np.load(dir_name + '/results.npy')
    # train_data, test_data = np.load(dir_name + '/data.npy')
    # os.remove(dir_name + '/data.npy')
    
    plot_func = PlotOneItem(args)
    plot_func.plot_vv(perform, data, dir_name)
    plot_func.plot_results(perform, data, dir_name)

################################################################################ 
# Main function  
################################################################################
if __name__ == '__main__':
    args = config_cl()
    
    dir_name = 'plots/' + str(args.num_agent) + 'agent_' + str(args.distribution_type)
                    
    if not os.path.exists(dir_name):
            os.makedirs(dir_name)
    
    main(args, dir_name)
    
    