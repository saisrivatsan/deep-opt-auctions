from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class OptRevOneBidder:
    def __init__(self, config, data):
        self.config = config
        self.data = data
    
    def opt_rev(self):
        num_agents = self.config.num_agents
        num_items = self.config.num_items
        sample_val = self.data
            
        num_instances = sample_val.shape[0]
        
        if self.config.agent_type == 'additive':
            if num_agents == 1 and num_items == 2:
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
                
            elif num_agents == 1 and num_items == 10:
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
            if num_agents == 1 and num_items == 2:
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
                
                
class OptRevMultiBidders:
    def __init__(self, config, data):
        self.config = config
        self.data = data
    
    def max_0(self, x):
        return max(x, 0)
    
    def AMA_vcg(self, val, alg):
        num_items = self.config.num_items
        num_agents = self.config.num_agents
            
        if self.config.distribution_type == 'uniform':
            allocation = np.array([[[0, 0], [0, 0]], [[1, 0], [0, 0]], [[0, 0], [0, 1]],
                                   [[1, 0], [0, 0]], [[0, 0], [1, 0]], [[1, 0], [0, 1]],
                                   [[0, 1], [1, 0]], [[1, 1], [0, 0]], [[0, 0], [1, 1]]])
            if alg == 'VVCA':
                Lambda = np.array([1.28, 0.66, 0.66, 0.72, 0.63, 0.09, 0, 0.39, 0.30])
                mu_1 = 1.0
                mu_2 = 1.09
            elif alg == 'AMA_bsym':
                Lambda = np.array([1.15, 0, 0, 0, 0, 0.05, 0.05, 0.33, 0.33])
                mu_1 = 1.0
                mu_2 = 1.0
            
            mu = np.array([[mu_1, mu_1], [mu_2, mu_2]])
            mu_copy = np.reshape(np.tile(mu, [9,1]), [9, num_agents, num_items])
            val_copy = np.reshape(np.tile(val, [9, 1]), [9, num_agents, num_items])
            
            social_welfare = np.sum(mu_copy * val_copy * allocation, axis = (1, 2)) + Lambda
            idx = np.argmax(social_welfare)
            opt_alloc = allocation[idx,:,:]
            opt_sw_agent1_opt_alloc = np.sum((mu * val * opt_alloc)[1,:]) + Lambda[idx]
            opt_sw_agent2_opt_alloc = np.sum((mu * val * opt_alloc)[0,:]) + Lambda[idx]
            #opt_sw = social_welfare[idx]
            

            sw_agent1 = np.sum((mu_copy * val_copy * allocation)[:,1,:], axis = 1) + Lambda
            sw_agent2 = np.sum((mu_copy * val_copy * allocation)[:,0,:], axis = 1) + Lambda
            opt_sw_agent1 = sw_agent1[np.argmax(sw_agent1)]
            opt_sw_agent2 = sw_agent2[np.argmax(sw_agent2)]


            payment_agent1 = (opt_sw_agent1 - opt_sw_agent1_opt_alloc)/mu_1
            payment_agent2 = (opt_sw_agent2 - opt_sw_agent2_opt_alloc)/mu_2
            

            return payment_agent1 + payment_agent2
        
        elif self.config.distribution_type == 'CA_sym_uniform_12':
            allocation = np.array([[[0,0,0], [0,0,0]], [[1,0,0], [0,0,0]], [[0,0,0], [0,1,0]],
                                   [[1,0,0], [0,0,0]], [[0,0,0], [1,0,0]], [[1,0,0], [0,1,0]],
                                   [[0,1,0], [1,0,0]], [[0,0,1], [0,0,0]], [[0,0,0], [0,0,1]]])
            if alg == 'VVCA':
                Lambda = np.array([3.50, 2.25, 1.88, 2.25, 1.88, 0.63, 0.63, 1.52, 0])
                mu_1 = 1.0
                mu_2 = 1.43
            elif alg == 'AMA_bsym':
                Lambda = np.array([2.95, 1.53, 1.53, 1.485, 1.48, 0.27, 0.27, 0.44, 0.44])
                mu_1 = 1.0
                mu_2 = 1.0
            
            mu = np.array([[mu_1, mu_1, mu_1], [mu_2, mu_2, mu_2]])
            mu_copy = np.reshape(np.tile(mu, [9, 1]), [9, num_agents, 3])
            val_copy = np.reshape(np.tile(val, [9, 1]), [9, num_agents, 3])
            
            social_welfare = np.sum(mu_copy * val_copy * allocation, axis = (1, 2)) + Lambda
            idx = np.argmax(social_welfare)
            opt_alloc = allocation[idx,:,:]
            opt_sw_agent1_opt_alloc = np.sum((mu * val * opt_alloc)[1,:]) + Lambda[idx]
            opt_sw_agent2_opt_alloc = np.sum((mu * val * opt_alloc)[0,:]) + Lambda[idx]
            #opt_sw = social_welfare[idx]
            

            sw_agent1 = np.sum((mu_copy * val_copy * allocation)[:,1,:], axis = 1) + Lambda
            sw_agent2 = np.sum((mu_copy * val_copy * allocation)[:,0,:], axis = 1) + Lambda
            opt_sw_agent1 = sw_agent1[np.argmax(sw_agent1)]
            opt_sw_agent2 = sw_agent2[np.argmax(sw_agent2)]


            payment_agent1 = (opt_sw_agent1 - opt_sw_agent1_opt_alloc)/mu_1
            payment_agent2 = (opt_sw_agent2 - opt_sw_agent2_opt_alloc)/mu_2
            
            return payment_agent1 + payment_agent2
            
        elif self.config.distribution_type == 'CA_asym_uniform_12_15':
            allocation = np.array([[[0,0,0], [0,0,0]], [[1,0,0], [0,0,0]], [[0,0,0], [0,1,0]],
                                   [[1,0,0], [0,0,0]], [[0,0,0], [1,0,0]], [[1,0,0], [0,1,0]],
                                   [[0,1,0], [1,0,0]], [[0,0,1], [0,0,0]], [[0,0,0], [0,0,1]]])
            if alg == 'VVCA':
                Lambda = np.array([3.88, 2.63, 1.38, 2.75, 1.25, 0.25, 0, 1.88, 0.06])
                mu_1 = 1.0
                mu_2 = 0.88
            elif alg == 'AMA_bsym':
                Lambda = np.array([5.14, 1.70, 1.70, 1.70, 1.70, 0.60, 0.60, 0.03, 0.03])
                mu_1 = 1.0
                mu_2 = 1.0
            
            mu = np.array([[mu_1, mu_1, mu_1], [mu_2, mu_2, mu_2]])
            mu_copy = np.reshape(np.tile(mu, [9, 1]), [9, num_agents, 3])
            val_copy = np.reshape(np.tile(val, [9, 1]), [9, num_agents, 3])
            
            social_welfare = np.sum(mu_copy * val_copy * allocation, axis = (1, 2)) + Lambda
            idx = np.argmax(social_welfare)
            opt_alloc = allocation[idx,:,:]
            opt_sw_agent1_opt_alloc = np.sum((mu * val * opt_alloc)[1,:]) + Lambda[idx]
            opt_sw_agent2_opt_alloc = np.sum((mu * val * opt_alloc)[0,:]) + Lambda[idx]
            #opt_sw = social_welfare[idx]
            

            sw_agent1 = np.sum((mu_copy * val_copy * allocation)[:,1,:], axis = 1) + Lambda
            sw_agent2 = np.sum((mu_copy * val_copy * allocation)[:,0,:], axis = 1) + Lambda
            opt_sw_agent1 = sw_agent1[np.argmax(sw_agent1)]
            opt_sw_agent2 = sw_agent2[np.argmax(sw_agent2)]


            payment_agent1 = (opt_sw_agent1 - opt_sw_agent1_opt_alloc)/mu_1
            payment_agent2 = (opt_sw_agent2 - opt_sw_agent2_opt_alloc)/mu_2
            
            return payment_agent1 + payment_agent2        
            
    def opt_rev(self):
        num_agents = self.config.num_agents
        num_items = self.config.num_items
        sample_val = self.data
        num_instances = sample_val.shape[0]
        
        if self.config.agent_type == 'additive':
            if self.config.distribution_type == 'uniform':
                if num_items == 2 and num_agents == 2:
                    rev_VVCA = 0.0
                    rev_AMA_bsym = 0.0
                    for i in range(num_instances):
                        rev_VVCA += self.AMA_vcg(sample_val[i,:,:], 'VVCA')
                        rev_AMA_bsym += self.AMA_vcg(sample_val[i,:,:], 'AMA_bsym')
                
                    return rev_VVCA/num_instances, rev_AMA_bsym/num_instances
                
                elif num_items >= 2 and num_agents >= 2:
                    rev_item_myerson = 0.0
                    for i in range(num_instances):
                        for j in range(num_items):
                            index = np.argmax(sample_val[i,:,j])
                            max_val = np.max(sample_val[i,:,j])
                            second_max_val = np.sort(sample_val[i,:,j])[-2]
                            if(max_val < 0.5):
                                rev_item_myerson += 0.0
                            elif(max_val >= 0.5 and second_max_val < 0.5):
                                rev_item_myerson += 0.5
                            else:
                                rev_item_myerson += second_max_val

                    rev_item_myerson = rev_item_myerson/num_instances
                    
                    return rev_item_myerson
                
                 
                    
            elif self.config.distribution_type == 'CA_sym_uniform_12':
                if num_items == 2 and num_agents == 2:
                    rev_VVCA = 0.0
                    rev_AMA_bsym = 0.0
                    for i in range(num_instances):
                        rev_VVCA += self.AMA_vcg(sample_val[i,:,:], 'VVCA')
                        rev_AMA_bsym += self.AMA_vcg(sample_val[i,:,:], 'AMA_bsym')
                
                    return rev_VVCA/num_instances, rev_AMA_bsym/num_instances
            
            elif self.config.distribution_type == 'CA_asym_uniform_12_15':
                if num_items == 2 and num_agents == 2:
                    rev_VVCA = 0.0
                    rev_AMA_bsym = 0.0
                    for i in range(num_instances):
                        rev_VVCA += self.AMA_vcg(sample_val[i,:,:], 'VVCA')
                        rev_AMA_bsym += self.AMA_vcg(sample_val[i,:,:], 'AMA_bsym')
                
                    return rev_VVCA/num_instances, rev_AMA_bsym/num_instances
            
                
            elif self.config.distribution_type == 'uniform_2supp':
                if num_items == 2:
                    n = num_agents
                    p = self.config.yao_prob[0]
                    a = self.config.yao_supp_a
                    b = self.config.yao_supp_b
                    p_0 = p ** (2 ** n)
                    p_1 = 2 * n * (p **(2*n - 1)) * (1-p)
                    p_2 = 2 * (p ** n) * (1 - p ** n - n * (p ** (n-1)) * (1-p))
                    r_D_theory = 2.0 * (1 - p ** n) * b\
                                 + p_0 * self.max_0(2*a - (b-a) * (1-p**2)/(p**2))\
                                 + p_1 * self.max_0(a - (1-p) * (b-a)/(2 * p))\
                                 + p_2 * self.max_0(a - (1-p)*(b-a)/p)
                                 
                    r_B_theory = 2.0 * (1 - p ** n) * b\
                                 + p_0 * self.max_0(2*a - (b-a) * (1-p**2)/(p**2))\
                                 + (p_1 + p_2) * self.max_0(a - (1-p) * (b-a)/(2 * p))
                        
                    v_1 = a * (1 + p**2)/(1 - p**2)
                    v_2 = a/(1 - p)
                    v_3 = a * (1 + p)/(1 - p)
                    
                    r_D_practice = 0
                    if b >= v_2 and b < v_3:
                        for i in range(num_instances):
                            binary_a_row = sample_val[i,:,:] == a
                            num_a_row = np.count_nonzero(np.sum(binary_a_row, axis=1) == 2)
                            if num_a_row == n:
                                r_D_practice += 0
                            elif num_a_row == (n-1):
                                r_D_practice += a + b
                            else:
                                if np.max(sample_val[i,:,0]) == b:
                                    r_D_practice += b
                                if np.max(sample_val[i,:,1]) == b:
                                    r_D_practice += b
                                    
                        r_D_practice = r_D_practice/num_instances
                                
                    return r_D_theory, r_B_theory, r_D_practice
            
            elif self.config.distribution_type == 'uniform_3supp':
                if num_items == 2:
                    n = num_agents
                    r_item_myerson = 0.0
                    r_optimal_bundling = 0.0
                    for i in range(num_instances):
                        for j in range(num_items):
                            x1 = sample_val[i, 0, j]
                            x2 = sample_val[i, 1, j]
                            max_val = max([x1, x2])
                            second_max_val = np.sort([x1, x2])[0]
                            if(max_val >= 1.0 and second_max_val >= 1.0):
                                r_item_myerson += second_max_val
                            elif(max_val >= 1.0 and second_max_val < 1.0):
                                r_item_myerson += 1.0
                            else:
                                r_item_myerson += 0.0
                    
                    r_item_myerson = r_item_myerson/num_instances
                    
                    for i in range(num_instances):
                        bundle_1 = sum(sample_val[i, 0, :])
                        bundle_2 = sum(sample_val[i, 1, :])
                        max_bundle = max([bundle_1, bundle_2])
                        second_max_bundle = np.sort([bundle_1, bundle_2])[0]
                        if(max_bundle >= 2.0 and second_max_bundle >= 2.0):
                            r_optimal_bundling += second_max_bundle
                        elif(max_bundle >= 2.0 and second_max_bundle < 2.0):
                            r_optimal_bundling += 2.0
                        else:
                            r_optimal_bundling += 0.0
                        
                    r_optimal_bundling = r_optimal_bundling/num_instances
                    
                    return r_item_myerson, r_optimal_bundling
                    
    def winner(self):
        sample_val, sample_misreport, navigator = self.data
        num_items = self.config.num_items
        num_agents = self.config.num_agents
        
        if self.config.distribution_type == 'uniform_2supp':
            if num_items == 2 and num_agents == 2:
                p = self.config.yao_prob[0]
                a = self.config.yao_supp_a
                b = self.config.yao_supp_b
                
                # Record num instances, agents, misreports
                num_instances = sample_val.shape[0]    
                n = num_agents
                
                win_index = np.zeros((num_instances, num_agents, num_items))
                v_1 = a * (1 + p**2)/(1 - p**2)
                v_2 = a/(1 - p)
                v_3 = a * (1 + p)/(1 - p)
                    
                if(b >= v_2 and b < v_3):
                    for i in range(num_instances):
                        binary_a_row = sample_val[i,:,:] == a
                        binary_both_a = np.sum(binary_a_row, axis=1) == 2
                        binary_no_both_a = np.sum(binary_a_row, axis=1) != 2
                        num_a_row = np.count_nonzero(binary_both_a)
                        if(num_a_row == n):
                            win_index[i,:,:] = np.zeros((2, 2))
                        elif(num_a_row == (n-1)):
                            win_index[i,:,:] = np.transpose(
                                               np.reshape(np.tile(binary_no_both_a, [2]), [2, 2]))
                        else:
                            if(np.max(sample_val[i,:,0]) == b):
                                win_index[i,:,0] =\
                                        (sample_val[i,:,0] == b)/np.sum(sample_val[i,:,0] == b)
                            if(np.max(sample_val[i,:,1]) == b):
                                win_index[i,:,1] =\
                                        (sample_val[i,:,1] == b)/np.sum(sample_val[i,:,1] == b)
                
                return win_index
            
            
def bundle_myserson(sample_val, rp):
    rev = 0.0
    num_instances = sample_val.shape[0]
    for i in range(num_instances):
        v = sample_val[i,:,:].sum(-1)
        index = np.argmax(v)
        max_val = np.max(v)
        second_max_val = np.sort(v)[-2]
        if(max_val < rp):
            rev += 0.0
        elif(max_val >= rp and second_max_val < rp):
            rev += rp
        else:
            rev += second_max_val

    rev = rev/num_instances
    return rev

class AscendingAuction:
    def __init__(self, config, data):
        self.config = config
        self.data = data
     
    def find_min_overdemand(self, val, p):
        num_agents = self.config.num_agents
        num_items = self.config.num_items
        
        demand = np.zeros((num_agents, num_items))
        matching = np.zeros((num_agents, num_items))
        
        for i in range(num_agents):
            demand[i,:] = ((val[i,:] - p) == np.max(val[i,:] - p)) * (p <= np.max(val[i,:])) * 1.0
        
        # g = Graph(demand)
        # num_match_item, match_item = g.maxBPM()
        
        min_overdemand = np.zeros(num_items)
        if(num_items == 2):        
            x_1 = ((demand[:,0] - demand[:,1]) > 0) * 1.0
            x_2 = ((demand[:,1] - demand[:,0]) > 0) * 1.0
            x_3 = (demand[:,0] == 1) * (demand[:,1] == 1) * 1.0
            if(np.sum(x_1) >= 2):
                min_overdemand[0] = 1.0
                min_overdemand[1] = 0
            elif(np.sum(x_2) >= 2):
                min_overdemand[0] = 0
                min_overdemand[1] = 1.0
            elif(np.sum(x_3) >= 3):
                min_overdemand[0] = 1.0
                min_overdemand[1] = 1.0
        
        return min_overdemand
        
            
    def compute_min_competitive_price(self, val):
        eps = 0.3
        num_agents = self.config.num_agents
        num_items = self.config.num_items
        p = np.zeros(num_items)
        minimal_overdemand = self.find_min_overdemand(val, p)
        
        while(np.sum(minimal_overdemand) >= 1e-4):
            p += minimal_overdemand * eps
            minimal_overdemand = self.find_min_overdemand(val, p)
        
        return p

    def ascending_rev(self, val):
        p = self.compute_min_competitive_price(val)
        num_agents = self.config.num_agents
        num_items = self.config.num_items
        demand = np.zeros((num_agents, num_items))
        
        revenue = 0.0
        for i in range(num_agents):
            demand[i,:] = ((val[i,:] - p) == np.max(val[i,:] - p)) * (p <= np.max(val[i,:])) * 1.0
            if(np.sum(demand[i,:]) == 0):
                revenue += 0.0
            else:
                revenue += np.sum(demand[i,:] * p)/np.sum(demand[i,:])
        
        return revenue
        
    def rev_compute_aa(self):
        sample_val = self.data
        num_instances = sample_val.shape[0]
        rev_ascending_auction = 0
        for i in range(num_instances):
            rev_ascending_auction += self.ascending_rev(sample_val[i,:,:])
        rev_ascending_auction = rev_ascending_auction/num_instances
        
        return rev_ascending_auction
    
