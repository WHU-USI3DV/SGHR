import torch
from tqdm import tqdm
import utils.utils as utils

# validation of group feature extraction
class build_validation:
    def __init__(self, cfg):
        self.cfg=cfg
        self.k = 10

    def _val_recall_topk(self, gt_overlap, pre_overlap):
        k = int(min(self.k, gt_overlap.shape[0]/2.0))
        n_scan = gt_overlap.shape[0]
        truely_pre = 0
        truely_gt  = k * n_scan
        for s in range(n_scan):
            gt_over_s = gt_overlap[s]
            pre_over_s = pre_overlap[s]
            # gt from large to small
            arg_gt  = torch.argsort(-gt_over_s)[0:k]
            # pre from large to small
            arg_pre = torch.argsort(-pre_over_s)[0:k]
            for i in arg_pre:
                if i in arg_gt:
                    truely_pre+=1
        return truely_pre / truely_gt

    def __call__(self, 
                 model, 
                 eval_dataset, 
                 loss_funcs):
        # evaluation mode
        model.eval()
        # conduct validation
        val_loss, val_recall = [], []
        for i, batch in enumerate(tqdm(eval_dataset)):
            batch = utils.to_cuda(batch)
            with torch.no_grad():
                output = model(batch)
                # calculate loss
                loss = loss_funcs(output)            
                val_loss.append(loss)
                # for validation recall
                pre_overlap = utils._f2overlap(output['vlad_gf'][-1])
                recall = self._val_recall_topk(output['gt_overlap'], pre_overlap)
                val_recall.append(recall)
        # val loss
        val_loss=torch.mean(torch.tensor(val_loss))
        # val recall
        val_recall=torch.mean(torch.tensor(val_recall))
        # train mode
        model.train()
        return {"val_loss":val_loss, "val_recall":val_recall}

