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

#asymmetric_uniform_daskalakis
if setting == "additive_1x2_uniform_416_47":
    cfg = additive_1x2_uniform_416_47_config.cfg
    Generator = uniform_416_47_generator.Generator

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
data = np.array([ next(generator.gen_func)[0] for _ in range(cfg.test.num_batches)])
data = data.reshape(-1, cfg.num_agents, cfg.num_items)

if setting == "unit_2x2_uniform":
    print(AscendingAuction(cfg, data).rev_compute_aa())
elif cfg.num_agents > 1: print(OptRevMultiBidders(cfg, data).opt_rev())
else: print(OptRevOneBidder(cfg, np.squeeze(data, 1)).opt_rev())
    

