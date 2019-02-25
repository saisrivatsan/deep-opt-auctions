from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import matplotlib
matplotlib.use('Agg')
import numpy as np
import pylab as pl
import sys
import os

from baseline.baseline import OptRevOneItem

class PlotOneItem():
    def __init__(self, args):
        self.args = args

    def plot_vv(self, performance_stats, data, dir_name):
        matplotlib.rcParams.update({'font.size': 9})
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            
        sample_val = data
        rev_recordings, alloc_error_recordings, vv_recordings = performance_stats
        
        num_agents = self.args.num_agent
        num_recordings = rev_recordings.shape[0]
        
        for i in range(num_agents):
            pl.figure(figsize=(4, 2.5))
            ss = np.sort(sample_val[:,i])
            vv = pl.plot(ss, np.sort(vv_recordings[:,i]), c = "b", linewidth = 2, linestyle = '-')
            vv_true = pl.plot(ss, np.sort(OptRevOneItem(self.args, data).compute_vv(sample_val[:,i], i)),\
                        c = "r", linewidth = 2, linestyle = '--')
            
            j=i+1
            pl.xlabel('$v_' + str(j) + '$')
            pl.ylabel('$\phi_'+str(j)+'(v_'+str(j)+')$')
            pl.title('Agent ' + str(j))
            
            pl.xlim((np.min(ss), np.max(ss)))
            pl.savefig(dir_name + '/vv_' + str(j) + '.pdf', bbox_inches = 'tight', pad_inches = 0.05)
        
        if self.args.distribution_type != 'asymmetric_uniform':
            pl.figure(figsize=(4, 2.5))
            ss_0 = np.sort(sample_val[:,0])
            pl.plot(ss_0, np.sort(OptRevOneItem(self.args, data).compute_vv(sample_val[:,0], 0)),\
                    'r', linewidth = 2, linestyle = '-', label = 'Myerson Transformation')
            color = ['b', 'g', 'm']
            line = ['--', '-.', ':']
            for i in range(num_agents):
                ss = np.sort(sample_val[:,i])
                pl.plot(ss, np.sort(vv_recordings[:,i]), c = color[i], linewidth = 2,\
                    linestyle = line[i], label = 'Agent ' + str(i+1))
                pl.xlim((np.min(ss), np.max(ss)))
            
            pl.legend(loc='best')            
            pl.xlabel('$v$')
            pl.ylabel('$\phi'+'(v'+')$')
            
            pl.savefig(dir_name + '/vv_total.pdf', bbox_inches = 'tight', pad_inches = 0.05)
            
            
    def plot_results(self, performance_stats, data, dir_name):
    
        skip_iter = self.args.skip_iter
        
        sample_val = data
        rev_recordings, alloc_error_recordings, vv_recordings = performance_stats
        
        num_agents = self.args.num_agent
        num_instances = sample_val.shape[0]
        num_recordings = rev_recordings.shape[0]
        
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        
        matplotlib.rcParams.update({'font.size': 9, 'axes.labelsize': 'x-large'})
        
        fpa_rev, spa_rev, opt_rev = OptRevOneItem(self.args, data).opt_rev()
        
        ######
        # Plot revenue
        ######
        fig = pl.figure(figsize = (4,2.5))
        
        pl.plot(np.arange(num_recordings)*int(skip_iter), spa_rev * np.ones(num_recordings), c='r',\
                label='SPA', linewidth = 2.5, linestyle = '--')
        pl.plot(np.arange(num_recordings)*int(skip_iter), opt_rev * np.ones(num_recordings), c='g',\
                label='Optimal Mechanism', linewidth = 2.5, linestyle = '-')
        pl.plot(np.arange(num_recordings)*int(skip_iter), rev_recordings, c = 'b',\
                label='MyersonNet', linewidth = 2.5)
        pl.legend(loc='best')
        
        if self.args.distribution_type == 'uniform':
            pl.ylim([0.4, 0.8])        
        elif self.args.distribution_type == 'exponential':
            pl.ylim([2, 3])
        elif self.args.distribution_type == 'irregular':
            pl.ylim([2, 3])
        elif self.args.distribution_type == 'asymmetric_uniform':
            pl.ylim([1.5, 3]) 
            
        pl.xlabel('No. of iterations')
        pl.ylabel('Test revenue')

        pl.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        pl.savefig(dir_name + '/revenue.pdf', bbox_inches = 'tight', pad_inches = 0.05)
        
        ######
        # Plot Allocation Error
        ######
        
        fig = pl.figure(figsize = (4,2.5))
        pl.plot(np.arange(num_recordings)*int(skip_iter), alloc_error_recordings, c = 'b', \
                       linewidth = 2.5)
        pl.legend(loc='best')
        pl.ylim([0, 0.5])

        pl.xlabel('No. of iterations')
        pl.ylabel('Test allocation error')

        pl.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        pl.savefig(dir_name + '/alloc-error.pdf', bbox_inches = 'tight', pad_inches = 0.05)
        
        ind = -1
        nnet_rev = np.round(rev_recordings[ind], 3)
        f_res = open(dir_name + '/result.txt', 'w')
        f_res.write('opt_rev:' + str(opt_rev) + '\r\n' +\
                'spa_rev:' + str(spa_rev) + '\r\n' +\
                'rev_nnet:' + str(nnet_rev) + '\r\n')
        f_res.close()
         
        