from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class OptRevOneBidder:
    def __init__(self, config, data):
        self.config = config
        self.data = data
    
    def opt_rev(self):
        num_items = self.config.num_items
        sample_val = self.data
            
        num_instances = sample_val.shape[0]
        
        if self.config.agent_type == 'additive':
            if num_items == 2:
                if self.config.distribution_type == 'uniform':
                    revenue = 0.0
                    for i in range(num_instances):
                        x1 = sample_val[i, 0]
                        x2 = sample_val[i, 1]
                        if(x1 + x2 <= (4.0 - np.sqrt(2.0))/3.0 and x1 <= 2.0/3.0\
                            and x2 <= 2.0/3.0):
                            revenue += 0
                        elif(x1 > 2.0/3.0 and x2 <= (2.0 - np.sqrt(2.0))/3.0):
                            revenue += 2.0/3.0
                        elif(x2 > 2.0/3.0 and x1 <= (2.0 - np.sqrt(2.0))/3.0):
                            revenue += 2.0/3.0
                        else:
                            revenue += (4.0 - np.sqrt(2.0))/3.0
                    revenue = revenue/num_instances
                    return(revenue)
                                    
                elif self.config.distribution_type == 'asymmetric_uniform_daskalakis':
                    revenue = 0.0
                    for i in range(num_instances):
                        x1 = sample_val[i, 0]
                        x2 = sample_val[i, 1]
                        if(x1 <= 8.0 and x2 <= -0.5 * x1 + 8.0):
                            revenue += 0
                        elif(x1 <= 8.0 and x2 > -0.5 * x1 + 8.0):
                            revenue += 8.0
                        elif(x1 > 8.0):
                            revenue += 12.0
                    revenue = revenue/num_instances
                    return(revenue)
                    
                elif self.config.distribution_type == 'uniform_triangle':
                    revenue = 0.0
                    for i in range(num_instances):
                        x1 = sample_val[i, 0]
                        x2 = sample_val[i, 1]
                        if x1 + x2 <= np.sqrt(1.0/3.0):
                            revenue += 0
                        else:
                            revenue += np.sqrt(1.0/3.0)
                            
                    revenue = revenue/num_instances
                    return(revenue)
                    
                elif self.config.distribution_type == 'asymmetric_uniform_04_03':
                    rev_bundle = 0.0
                    rev_item = 0.0
                    for i in range(num_instances):
                        x1 = sample_val[i, 0]
                        x2 = sample_val[i, 1]
                        if x1 + x2 <= 2 * np.sqrt(2):
                            rev_bundle += 0
                        else:
                            rev_bundle += 2 * np.sqrt(2)
                    
                    for i in range(num_instances):
                        x1 = sample_val[i, 0]
                        x2 = sample_val[i, 1]
                        if x1 >= 2.0:
                            rev_item += 2.0
                        if x2 > 1.5:
                            rev_item += 1.5
                            
                    rev_bundle = rev_bundle/num_instances
                    rev_item = rev_item/num_instances
                    return rev_bundle, rev_item
                
                elif self.config.distribution_type == 'asymmetric_uniform_14_13':
                    rev_bundle = 0.0
                    rev_item = 0.0
                    for i in range(num_instances):
                        x1 = sample_val[i, 0]
                        x2 = sample_val[i, 1]
                        if x1 + x2 <= (4.0 + 2.0 * np.sqrt(10))/3.0:
                            rev_bundle += 0
                        else:
                            rev_bundle += (4.0 + 2.0 * np.sqrt(10))/3.0
                    
                    for i in range(num_instances):
                        x1 = sample_val[i, 0]
                        x2 = sample_val[i, 1]
                        if x1 >= 2.0:
                            rev_item += 2.0
                        if x2 > 1.5:
                            rev_item += 1.5
                            
                    rev_bundle = rev_bundle/num_instances
                    rev_item = rev_item/num_instances
                    return rev_bundle, rev_item
                
            elif num_items == 10:
                if self.config.distribution_type == 'uniform':
                    rev_myerson = 0.0
                    rev_bundle = 0.0
                    for i in range(num_instances):
                        for j in range(num_items):
                            if(sample_val[i,j] >= 0.5):
                                rev_myerson += 0.5
                    rev_myerson = rev_myerson/num_instances
                    
                    for i in range(num_instances):
                        if(sum(sample_val[i,:]) >= 3.9292):
                            rev_bundle += 3.9292
                    rev_bundle = rev_bundle/num_instances
                    
                    return(rev_myerson, rev_bundle)

        elif self.config.agent_type == 'unit_demand':
            if num_items == 2:
                if self.config.distribution_type == 'uniform':
                    revenue = 0.0
                    for i in range(num_instances):
                        x1 = sample_val[i, 0]
                        x2 = sample_val[i, 1]
                        if(x1  <= np.sqrt(3.0)/3.0 and x2 <= np.sqrt(3.0)/3.0):
                            revenue += 0
                        elif(x1 > np.sqrt(3.0)/3.0 and x1 >= x2):
                            revenue += np.sqrt(3.0)/3.0
                        elif(x2 > np.sqrt(3.0)/3.0 and x1 <= x2):
                            revenue += np.sqrt(3.0)/3.0
                    revenue = revenue/num_instances
                    return(revenue)
            
            if self.config.distribution_type == 'uniform_23':
                revenue = 0.0
                for i in range(num_instances):
                    x1 = sample_val[i, 0]
                    x2 = sample_val[i, 1]
                    t1 = 4.0/3.0 + np.sqrt(4.0 + 3.0/2.0)/3.0 # Payment threshold
                    if(x1 + x2 <= 2.0 * t1):
                        revenue += 0
                    elif(x1 >= x2 + 1.0/3.0):
                        revenue += 1.0/6.0 + t1
                    elif(x2 >= x1 + 1.0/3.0):
                        revenue += 1.0/6.0 + t1
                    else:
                        revenue += t1
                revenue = revenue/num_instances
                return(revenue)
