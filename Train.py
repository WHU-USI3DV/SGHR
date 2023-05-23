import os
import time
from train.trainer import Trainer
import utils.parses as parses

cfg,_ = parses.get_config()

# backup the existing checkpoint
if os.path.exists(cfg.model_fn):
    now_clock = time.strftime('%Y%m%d_%H%M%S')
    modi_model_fn = f'{cfg.model_fn}_backup_{now_clock}'
    print(f'The existing checkpoint folder {cfg.model_fn} will be saved to {modi_model_fn}')
    os.system(f'mv {cfg.model_fn} {modi_model_fn}')

# train
generator = Trainer(cfg)
generator.run()
    