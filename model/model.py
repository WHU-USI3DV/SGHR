import torch
import torch.nn as nn
from model.netvlad import NetVLAD
 
class GF_VLAD(nn.Module):
    def __init__(self, cfg, normalize = True):
        super(GF_VLAD, self).__init__()
        self.drop = nn.Dropout(cfg.drop_out)
        self.pool = NetVLAD(num_clusters=cfg.vlad_cluster, dim=cfg.vlad_dim)
        self.normalize = normalize

    def forward(self, data_dict):
        # load local descriptors
        feats_list = data_dict['feats']
        # drop out the input features
        feats_list = [self.drop(feat) for feat in feats_list]
        # extract global features
        gf_list = [self.pool(feat) for feat in feats_list]
        # global features not normalized
        feats = torch.cat(gf_list, dim=0)
        # normalize features
        if self.normalize:
            feats = feats / torch.norm(feats, dim = 1, keepdim=True)
        return feats

class VLAD_MLP(nn.Module):
    def __init__(self, cfg, heads = 1):
        super(VLAD_MLP, self).__init__()
        self.cfg = cfg
        self.vlador = GF_VLAD(cfg, normalize=True)
    
    def forward(self, data_dict):
        gfs = []
        # conduct vlad for global feature extraction (already normalized)
        gf = self.vlador(data_dict)
        gfs.append(gf)
        return {
            'vlad_gf': gfs,
            'gt_overlap': data_dict['gt_overlap']
        }
    
              
name2model = {
    'vlad':VLAD_MLP,
}