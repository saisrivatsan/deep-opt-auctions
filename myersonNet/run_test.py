from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import tensorflow as tf


from nets import *
from cfgs import *
from data import *
from trainer import *

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
    
net = net.Net(cfg, "test")
generator = Generator(cfg, "test")
m = trainer.Trainer(cfg, "test", net)
m.test(generator)
