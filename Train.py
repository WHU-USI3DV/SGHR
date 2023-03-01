from train import Trainer
import utils.parses as parses

cfg,_ = parses.get_config()
generator = Trainer(cfg)
generator.run()
    