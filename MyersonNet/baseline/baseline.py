from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

class OptRevOneItem:
    def __init__(self, config, data):
        self.config = config
        self.data = data
    
    '''
    Compute the virtual value given the value and the bidder id
    '''
    def compute_vv(self, v, i):
        distr_type = self.config.distribution_type
        if distr_type == 'uniform':
            return(2.0 * v - 1.0)
        elif distr_type == 'irregular':
            n = len(v)
            vv = np.zeros(n)
            for i in range(n):
                if(v[i] <= (7.0 - np.sqrt(5.0))/2.0):
                    vv[i] = 2.0 * v[i] - 4.0
                elif(v[i] <= (11.0 - np.sqrt(5.0))/2.0):
                    vv[i] = 3.0 - np.sqrt(5.0)
                else:
                    vv[i] = 2.0 * v[i] - 8.0
            return(vv)
        elif distr_type == 'exponential':
            return(v - 3.0)
        elif distr_type == 'asymmetric_uniform':
            return(2.0 * v - (i+1))
            
    def compute_vv_inv(self, x):
        distr_type = self.config.distribution_type
        if distr_type == 'uniform':
            n = len(x)
            max_val = np.max(x)
            second_max_val = np.sort(x)[-2]
            if(max_val < 0):
                return(0)
            elif(max_val >=0 and second_max_val < 0):
                return(0.5)
            else:
                return((second_max_val + 1.0)/2.0)
        elif distr_type == 'irregular':
            max_val = np.max(x)
            second_max_val = np.sort(x)[-2]
            if max_val < 0:
                return 0
            elif max_val <= 3.0 - np.sqrt(5.0) and second_max_val < 0:
                return 2.0
            elif max_val <= 3.0 - np.sqrt(5.0) and second_max_val <= 3.0 - np.sqrt(5.0):
                return (second_max_val + 4.0)/2.0
            elif max_val > 3.0 - np.sqrt(5.0) and second_max_val < 0:
                return 2.0
            elif max_val > 3.0 - np.sqrt(5.0) and second_max_val < 3.0 - np.sqrt(5.0):
                return (second_max_val + 4.0)/2.0
            elif max_val > 3.0 - np.sqrt(5.0) and second_max_val == 3.0 - np.sqrt(5.0):
                return (11.0 - np.sqrt(5.0))/2.0 - 2.0/(sum(x == second_max_val)+1.0)
            else:
                return (second_max_val + 8.0)/2.0
        elif distr_type == 'exponential':
            n = len(x)
            max_val = np.max(x)
            second_max_val = np.sort(x)[-2]
            if(max_val < 0):
                return(0)
            elif(max_val >=0 and second_max_val < 0):
                return(3.0)
            else:
                return(second_max_val + 3.0)
        elif distr_type == 'asymmetric_uniform':
            n = len(x)
            max_val = np.max(x)
            second_max_val = np.sort(x)[-2]
            index = np.argmax(x)
            if(max_val < 0):
                return(0)
            elif(max_val >=0 and second_max_val < 0):
                return((index + 1.0)/2.0)
            else:
                return((second_max_val + index + 1.0)/2.0)
    
    def opt_rev(self):
        # read data
        sample_val = self.data

        # Record num instances, agents, misreports
        num_instances = sample_val.shape[0]
        num_agent = sample_val.shape[1]
        distr_type = self.config.distribution_type
        
        fpa_revenue = np.mean(np.amax(sample_val, axis=1)) 
        spa_revenue = np.mean(np.sort(sample_val, axis=1)[:,-2])
        
        if distr_type == 'uniform':
            virtual_val = 2.0 * sample_val - 1.0
            myerson_revenue = 0.0
            for i in range(num_instances):
                index = np.argmax(virtual_val[i,:])
                max_val = np.max(virtual_val[i,:])
                second_max_val = np.sort(virtual_val[i,:])[-2]
                if(max_val < 0):
                    myerson_revenue += 0
                elif(max_val >= 0 and second_max_val < 0):
                    myerson_revenue += 0.5
                else:
                    myerson_revenue += (second_max_val + 1.0)/2.0

            myerson_revenue = myerson_revenue/num_instances
            
        elif distr_type == 'irregular':
            virtual_val = np.zeros((num_instances, num_agent))
            for i in range(num_agent):          
                virtual_val[:,i] = self.compute_vv(sample_val[:,i], i)

            myerson_revenue = 0
            for i in range(num_instances):
                index = np.argmax(virtual_val[i,:])
                max_val = np.max(virtual_val[i,:])
                second_max_val = np.sort(virtual_val[i,:])[-2]
                if(max_val < 0):
                    myerson_revenue += 0
                elif(max_val >= 0 and second_max_val < 0):               
                    myerson_revenue += 2.0
                else:
                    myerson_revenue += self.compute_vv_inv(virtual_val[i,:])

            myerson_revenue = myerson_revenue/num_instances
            
        elif distr_type == 'exponential':
            virtual_val = np.zeros((num_instances, num_agent))
            for i in range(num_agent):          
                virtual_val[:,i] = self.compute_vv(sample_val[:,i], i)

            myerson_revenue = 0
            for i in range(num_instances):
                    myerson_revenue += self.compute_vv_inv(virtual_val[i,:])

            myerson_revenue = myerson_revenue/num_instances
            
        elif distr_type == 'asymmetric_uniform':
            virtual_val = np.zeros((num_instances, num_agent))
            for i in range(num_agent):          
                virtual_val[:,i] = self.compute_vv(sample_val[:,i], i)

            myerson_revenue = 0
            for i in range(num_instances):
                    myerson_revenue += self.compute_vv_inv(virtual_val[i,:])

            myerson_revenue = myerson_revenue/num_instances
            
        return fpa_revenue, spa_revenue, myerson_revenue  
            
    def winner(self):
        sample_val = self.data
        distr_type = self.config.distribution_type
        num_instances = sample_val.shape[0]
        num_agent = sample_val.shape[1]
        
        if distr_type == 'uniform':
            win_index = np.zeros((num_instances, num_agent))
            virtual_val = 2.0 * sample_val - 1.0
            for i in range(num_instances):
                index = np.argmax(virtual_val[i,:])
                if(np.max(virtual_val[i,:]) >= 0):
                    win_index[i, index] = 1  # Record the winner              
                    
        elif distr_type == 'irregular':
            win_index = np.zeros((num_instances, num_agent))
            virtual_val = np.zeros((num_instances, num_agent))
            for i in range(num_agent):
                virtual_val[:,i] = self.compute_vv(sample_val[:,i], i)

            for i in range(num_instances):
                max_val_vv = np.max(virtual_val[i,:])
                num_winner = sum(virtual_val[i,:] == max_val_vv)
                if(max_val_vv >= 0):
                    for j in range(num_agent):
                        if virtual_val[i,j] == max_val_vv:
                            win_index[i, j] = 1.0/num_winner  # Record the winner   
                            
        elif distr_type == 'exponential':
            win_index = np.zeros((num_instances, num_agent))
            virtual_val = np.zeros((num_instances, num_agent))
            for i in range(num_agent):
                virtual_val[:,i] = self.compute_vv(sample_val[:,i], i)

            for i in range(num_instances):
                max_val_vv = np.max(virtual_val[i,:])
                index = np.argmax(virtual_val[i,:])
                if(max_val_vv >= 0):
                    win_index[i, index] = 1 # Record the winner   
                    
        elif distr_type == 'asymmetric_uniform':
            win_index = np.zeros((num_instances, num_agent))
            b = np.tile(np.array([-1.0, -2.0, -3.0, -4.0, -5.0]), [num_instances, 1])
            virtual_val = 2 * sample_val + b
            for i in range(num_instances):
                index = np.argmax(virtual_val[i,:])
                if(np.max(virtual_val[i,:]) >= 0):
                    win_index[i, index] = 1  # Record the winner        
            
        return win_index  