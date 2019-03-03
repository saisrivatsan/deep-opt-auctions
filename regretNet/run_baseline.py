from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import tensorflow as tf


from nets import *
from cfgs import *
from data import *
from clip_ops.clip_ops import *
from trainer import *
from baseline.baseline import *

print("Setting: %s"%(sys.argv[1]))
setting = sys.argv[1]

if setting == "additive_1x2_uniform":
    cfg = additive_1x2_uniform_config.cfg
    Generator = uniform_01_generator.Generator
    print("OPT: ")

elif setting == "unit_1x2_uniform_23":
    cfg = unit_1x2_uniform_23_config.cfg
    Generator = uniform_23_generator.Generator
    print("OPT: ")

elif setting == "additive_2x2_uniform":
    cfg = additive_2x2_uniform_config.cfg
    Generator = uniform_01_generator.Generator
    print("(VVCA, AMA): ")

elif setting == "additive_2x3_uniform":
    cfg = additive_2x3_uniform_config.cfg
    Generator = uniform_01_generator.Generator
    print("(I-My, B-My): ")

elif setting == "additive_3x10_uniform":
    cfg = additive_3x10_uniform_config.cfg
    Generator = uniform_01_generator.Generator
    print("(I-My, B-My): ")
    
elif setting == "additive_5x10_uniform":
    cfg = additive_5x10_uniform_config.cfg
    Generator = uniform_01_generator.Generator
    print("(I-My, B-My): ")

elif setting == "CA_asym_uniform_12_15":
    cfg = CA_asym_uniform_12_15_config.cfg
    Generator = CA_asym_uniform_12_15_generator.Generator
    print("(VVCA, AMA): ")

elif setting == "CA_sym_uniform_12":
    cfg = CA_sym_uniform_12_config.cfg
    Generator = CA_sym_uniform_12_generator.Generator
    print("(VVCA, AMA): ")

elif setting == "additive_1x2_uniform_416_47":
    cfg = additive_1x2_uniform_416_47_config.cfg
    Generator = uniform_416_47_generator.Generator
    print("OPT: ")

elif setting == "additive_1x2_uniform_triangle":
    cfg = additive_1x2_uniform_triangle_config.cfg
    Generator = uniform_triangle_01_generator.Generator
    print("OPT: ")
    
elif setting == "unit_1x2_uniform":
    cfg = unit_1x2_uniform_config.cfg
    Generator = uniform_01_generator.Generator
    print("OPT: ")

elif setting == "additive_1x10_uniform":
    cfg = additive_1x10_uniform_config.cfg
    Generator = uniform_01_generator.Generator
    print("(I-My, B-My): ")
    
elif setting == "additive_1x2_uniform_04_03":
    cfg = additive_1x2_uniform_04_03_config.cfg
    Generator = uniform_04_03_generator.Generator
    print("(I-My, B-My): ")

elif setting == "unit_2x2_uniform":
    cfg = unit_2x2_uniform_config.cfg
    Generator = uniform_01_generator.Generator
    print("AA: ")
    
else:
    print("None selected")
    sys.exit(0)

np.random.seed(cfg.test.seed)
generator = Generator(cfg, 'test')

if not setting.startswith("CA"):
    data = np.array([ next(generator.gen_func)[0] for _ in range(cfg.test.num_batches)])
    data = data.reshape(-1, cfg.num_agents, cfg.num_items)
else:
    data = []
    c = []
    for i in range(cfg.test.num_batches):
        X, ADV, C, perm = next(generator.gen_func)
        data.append(X)
        c.append(C)

    data = np.array(data).reshape(-1, cfg.num_agents, cfg.num_items)
    c = np.array(c).reshape(-1, cfg.num_agents)
    x_bundle = np.sum(data, -1) + c
    x_in = np.zeros((cfg.test.num_instances, cfg.num_agents, cfg.num_items + 1))
    x_in[:,:,:cfg.num_items] = data
    x_in[:,:,-1] = x_bundle    
    data = x_in


if setting == "unit_2x2_uniform":
    print(AscendingAuction(cfg, data).rev_compute_aa())
elif cfg.num_agents > 1: print(OptRevMultiBidders(cfg, data).opt_rev())
else: print(OptRevOneBidder(cfg, np.squeeze(data, 1)).opt_rev())
   
if setting == "additive_3x10_uniform" or setting == "additive_5x10_uniform":
    print(bundle_myserson(data, rp = 3.92916))
if setting == "additive_2x3_uniform":
    print(bundle_myserson(data, rp = 1.16291)) 

