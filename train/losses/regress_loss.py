import torch
from train.losses.base import Loss, _f2overlap
    
class L1_loss(Loss):
    def __init__(self,cfg,topk = False):
        super().__init__()
        self.topk = topk
        self.loss=torch.nn.SmoothL1Loss(reduction='mean')
    
    def __call__(self,output):
        f_attn = torch.squeeze(output['vlad_gf'])
        gt = torch.squeeze(output['gt_overlap'])
        pre_attn = _f2overlap(f_attn, mode = 'l2')
        return self.loss(pre_attn,gt)