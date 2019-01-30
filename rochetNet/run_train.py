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


if setting == "additive_1x2_uniform":
    cfg = additive_1x2_uniform_config.cfg
    Net = additive_net.Net
    Generator = uniform_01_generator.Generator
    Trainer = trainer.Trainer
    
elif setting == "additive_1x2_uniform_416_47":
    cfg = additive_1x2_uniform_416_47_config.cfg
    Net = additive_net.Net
    Generator = uniform_416_47_generator.Generator
    Trainer = trainer.Trainer
    
elif setting == "additive_1x2_uniform_04_03":
    cfg = additive_1x2_uniform_04_03_config.cfg
    Net = additive_net.Net
    Generator = uniform_04_03_generator.Generator
    Trainer = trainer.Trainer
    
elif setting == "additive_1x2_uniform_triangle":
    cfg = additive_1x2_uniform_triangle_config.cfg
    Net = additive_net.Net
    Generator = uniform_triangle_01_generator.Generator
    Trainer = trainer.Trainer

elif setting == "additive_1x10_uniform":
    cfg = additive_1x10_uniform_config.cfg
    Net = additive_net.Net
    Generator = uniform_01_generator.Generator
    Trainer = trainer.Trainer

elif setting == "unit_1x2_uniform":
    cfg = unit_1x2_uniform_config.cfg
    Net = unit_net.Net
    Generator = uniform_01_generator.Generator
    Trainer = trainer.Trainer
     
elif setting == "unit_1x2_uniform_23":
    cfg = unit_1x2_uniform_23_config.cfg
    Net = unit_net.Net
    Generator = uniform_23_generator.Generator
    Trainer = trainer.Trainer
    
else:
    print("None selected")
    sys.exit(0)
    

net = Net(cfg, "train")
generator = [Generator(cfg, 'train'), Generator(cfg, 'val')]
m = Trainer(cfg, "train", net)
m.train(generator)
