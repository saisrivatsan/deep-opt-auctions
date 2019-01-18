from cfgs.config_3x10 import cfg
from trainer.uniform_additive import Trainer

m = Trainer(cfg, "train")
m.train()