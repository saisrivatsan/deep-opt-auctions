import numpy as np
from cfgs.config import cfg
from data.uniform_01_generator import Generator
from baseline.baseline import *

np.random.seed(cfg.test.seed)
gen = Generator(mode = "test", config = cfg)

data = np.array([ next(gen.gen_func)[0] for _ in range(cfg.test.num_batches)])
data = data.reshape(-1, cfg.num_agents, cfg.num_items)
print("DEBUG: xsum: %f"%(data.sum()) + " xshape: " + str(data.shape))

if cfg.distribution_type == "uniform" and cfg.agent_type == "additive":
    print("Revenue (Item-wise Myerson): "),
else:    
    print("Revenue (Baseline): "),
if cfg.num_agents > 1: print(OptRevMultiBidders(cfg, data).opt_rev())
else: print(OptRevOneBidder(cfg, np.squeeze(data, 1)).opt_rev())

if cfg.distribution_type == "uniform" and cfg.agent_type == "additive":
    """ Reserve price rp computed using Irwin-Hall distribution """    
    if cfg.num_items == 10:
        print("Revenue (Bundle Myerson): "),
        print(bundle_myserson(data, rp = 3.92916))
    elif cfg.num_items == 3:
        print("Revenue (Bundle Myerson): "),
        print(bundle_myserson(data, rp = 1.16291)) 
