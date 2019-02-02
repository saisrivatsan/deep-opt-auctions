from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import tensorflow as tf

from cfgs import *
from data import *
from baseline.baseline import *

print("Setting: %s"%(sys.argv[1]))
setting = sys.argv[1]

if setting == "uniform":
    cfg = m_3x1_sym_uniform_01_config.cfg
    Generator = sym_uniform_01.Generator
    
elif setting == "asymmetric_uniform":
    cfg = m_5x1_asym_uniform_config.cfg
    Generator = asym_uniform.Generator
    
elif setting == "exponential":
    cfg = m_3x1_exp_3_config.cfg
    Generator = exp_3.Generator
    
elif setting == "irregular":
    cfg = m_3x1_irregular_config.cfg
    Generator = irregular_03_38.Generator
    
else:
    print("None selected")
    sys.exit(0)
    
np.random.seed(cfg.test.seed)
generator = Generator(cfg, 'test')

data = np.array([ next(generator.gen_func) for _ in range(cfg.test.num_batches)])
data = data.reshape(-1, cfg.num_agents)
print(OptRevOneItem(cfg, data).opt_rev())
