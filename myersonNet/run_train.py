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


elif setting == "3x1_sym_uniform_01":
    cfg = m_3x1_sym_uniform_01_config.cfg
    Generator = uniform_triangle_01_generator.Generator
    
elif setting == "5x1_asym_uniform":
    cfg = m_5x1_asym_uniform_config.cfg
    Generator = uniform_triangle_01_generator.Generator
    
elif setting == "3x1_exp_3":
    cfg = m_3x1_exp_3_config.cfg
    Generator = uniform_triangle_01_generator.Generator
    
elif setting == "m_3x1_irregular_config":
    cfg = additive_1x2_uniform_triangle_config.cfg
    Generator = uniform_triangle_01_generator.Generator
    
else:
    print("None selected")
    sys.exit(0)
    
Net = net.Net(cfg, "train")
generator = [Generator(cfg, 'train'), Generator(cfg, 'val')]
m = trainer.Trainer(cfg, "train", net)
m.train(generator)
