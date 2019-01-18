from cfgs.config import cfg
from trainer.uniform_additive import Trainer

m = Trainer(cfg, "train")
m.train()