from dataops.dataset import (
    scenewisedataset
)
from torch.utils.data import DataLoader

def remove_first_channle(data_dicts):
    r"""
        output results without additionally add the first channel
    """
    batch_size = len(data_dicts)
    return data_dicts[0]
    
def bulid_trainvalset(cfg):
    # trainset
    tset = scenewisedataset(cfg, stage='train',point_limit=5000)
    tsloader = DataLoader(tset,
                      cfg.batch_size,
                      shuffle=True,
                      num_workers=cfg.worker_num,
                      collate_fn=remove_first_channle)
    # valset
    vset = scenewisedataset(cfg, stage='val',point_limit=5000)
    vsloader = DataLoader(vset,
                    cfg.batch_size_val,
                    shuffle=False,
                    num_workers=cfg.worker_num,
                    collate_fn=remove_first_channle)
    return tset, tsloader, vset, vsloader

def bulid_testset(cfg):
    # trainset
    tset = scenewisedataset(cfg, stage='test')
    tsloader = DataLoader(tset,
                      batch_size=1,
                      shuffle=False,
                      drop_last=False,
                      num_workers=cfg.worker_num,
                      collate_fn=remove_first_channle)
    return tset, tsloader