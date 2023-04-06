import torch
from train.losses.regress_loss import L1_loss

name2loss = {
    'l1': L1_loss
}

class bulid_loss_stack(torch.nn.Module):
    def __init__(self, cfg):
        self.losses = []
        self.types = cfg.loss_type
        self.weights = cfg.loss_weights
        for loss_name in self.types:
            self.losses.append(name2loss[loss_name](cfg))
    def __call__(self, output_dict):
        fs = output_dict['vlad_gf']
        gt = output_dict['gt_overlap']
        result = 0
        for i, loss in enumerate(self.losses):
            for f in fs:
                split = {
                    'vlad_gf':f,
                    'gt_overlap':gt
                }
                loss_item = loss(split)
                result += loss_item*self.weights[i]
        return result