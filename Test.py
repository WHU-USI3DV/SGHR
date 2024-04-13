import os,torch
import numpy as np
from tqdm import tqdm
from utils.parses import get_config
from model.model import name2model
from dataops import bulid_testset
from utils.utils import (to_cuda, 
 read_pickle,
 _f2overlap,
 make_non_exists_dir)

class tester():
    def __init__(self, cfg):
        self.cfg = cfg
        # for model
        self.pth_fn = f'{self.cfg.model_fn}/model_best.pth'
        self._init_load_model()

    def _init_load_model(self):
        self.network = name2model[self.cfg.model_type](self.cfg).cuda()
        self.network.eval()
        if os.path.exists(self.pth_fn): 
            checkpoint=torch.load(self.pth_fn)
            self.network.load_state_dict(checkpoint['network_state_dict'])
            # print(f'load model from {self.pth_fn}')
    
    def _generate_simmat(self, batch):
        batch = to_cuda(batch)
        with torch.no_grad():
            output = self.network(batch)
        gfs = output['vlad_gf']
        simmat = _f2overlap(gfs[-1],mode = 'l2')
        return simmat.cpu().numpy()

    def __call__(self):
        recalls = []
        # name only  --  change testset in cfg
        self.dset_list = read_pickle(self.cfg.testlist)
        # load descriptors
        _, self.dset = bulid_testset(cfg)
        # for saving
        self.d_save = f'{self.cfg.save_dir}/predict_overlap/{self.cfg.testset}'
        # estimation
        for i, batch in enumerate(tqdm(self.dset)):
            sn, _ = self.dset_list[i]
            save_fn = f'{self.d_save}/{sn}/ratio.npy'
            make_non_exists_dir(f'{self.d_save}/{sn}')
            simmat = self._generate_simmat(batch)
            np.save(save_fn, simmat)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',default='3dmatch',type=str,help='dataset name')
    args = parser.parse_args()
    
    testlists = {'3dmatch': './train/pkls/test_3dmatch.pkl',
                 'scannet': './train/pkls/test_scannet.pkl',
                 'ETH':     './train/pkls/test_eth.pkl'}
    
    cfg,_ = get_config()
    cfg.testset = args.dataset
    cfg.testlist = testlists[cfg.testset]
    
    test_runner = tester(cfg)
    test_runner()