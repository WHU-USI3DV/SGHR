import os, sys
# sys.path.append('..')
import torch
import numpy as np
from tqdm import tqdm
import utils.utils as utils
from torch.optim import Adam
# for training
from model.model import name2model
from dataops import bulid_trainvalset
from train import (
    bulid_loss_stack,
    build_validation
)

class Trainer:
    def __init__(self,cfg):
        self.cfg=cfg
        # hyperparameters
        self._load_hypers()
        # dataset
        _, self.train_set, _, self.val_set = bulid_trainvalset(cfg)
        print(f'train set len {len(self.train_set)}')
        print(f'val set len {len(self.val_set)}')
        # loss function
        self.loss_funcs = bulid_loss_stack(cfg)
        # validation
        self.val_evaluator = build_validation(cfg)
        # model and optimizer
        self.model_dir=f'{self.cfg.model_fn}'
        if not os.path.exists(self.model_dir): os.makedirs(self.model_dir)
        self.pth_fn=os.path.join(self.model_dir,'model.pth')
        self.best_pth_fn=os.path.join(self.model_dir,'model_best.pth')
        self._init_network()
        # logging
        self._init_logger()
        # for model save
        self.val_index = 'val_recall'
        self.greater_sign = 1 # =-1 smaller better, =1 greater better
        if self.greater_sign > 0:
            self.best_para = 0
        else:
            self.best_para = 1000
        self.start_step = 0
        
    def _load_hypers(self):
        self.batch_size = self.cfg.batch_size
        self.batch_size_val = self.cfg.batch_size_val
        self.worker_num = self.cfg.worker_num
        self.lr_init = self.cfg.lr_init
        self.lr_decay_rate = self.cfg.lr_decay_rate
        self.lr_decay_step = self.cfg.lr_decay_step
        self.epochs = self.cfg.epochs
        self.multi_gpus = self.cfg.multi_gpus
        self.train_log_step = self.cfg.train_log_step
        self.val_interval = self.cfg.val_interval
        self.save_interval = self.cfg.save_interval
        
    def _init_network(self):
        self.network = name2model[self.cfg.model_type](self.cfg).cuda()
        self.optimizer = Adam(filter(lambda p: p.requires_grad, self.network.parameters()), lr=self.lr_init)
        self.lr_setter=utils.ExpDecayLR(self.lr_init, self.lr_decay_rate, len(self.train_set)*self.lr_decay_step)
        
    def _load_model(self, strict = True):
        # if not os.path.exists(self.best_pth_fn):
        #     strict=False
        if os.path.exists(self.pth_fn): 
            checkpoint=torch.load(self.pth_fn)
            self.network.load_state_dict(checkpoint['network_state_dict'], strict=strict)
            if strict:
                self.best_para = checkpoint['best_para']
                self.start_step = checkpoint['step']
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f'==> resuming from step {self.start_step} best para {self.best_para}')

    def _save_model(self, step, save_fn=None):
        save_fn=self.pth_fn if save_fn is None else save_fn
        torch.save({
            'step':step,
            'best_para':self.best_para,
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        },save_fn)
    
    def _init_logger(self):
        self.logger = utils.Logger(self.model_dir)

    def _log_data(self,results,step,prefix='train',verbose=False):
        log_results={}
        for k, v in results.items():
            if isinstance(v,float) or np.isscalar(v):
                log_results[k] = v
            elif type(v)==np.ndarray:
                log_results[k]=np.mean(v)
            else:
                log_results[k]=np.mean(v.detach().cpu().numpy())
        self.logger.log(log_results,prefix,step,verbose)

    def run(self):
        self._load_model()
        pbar=tqdm(total=self.epochs*len(self.train_set),bar_format='{r_bar}')
        pbar.update(self.start_step)
        step=self.start_step
        wholeloss=0
        start_epoch=self.start_step//len(self.train_set)
        self.start_step=self.start_step-start_epoch*len(self.train_set)
        whole_step=len(self.train_set)*self.epochs
        for epoch in range(start_epoch,self.epochs):
            for i,train_data in enumerate(self.train_set):
                step+=1
                if not self.multi_gpus:
                    train_data = utils.to_cuda(train_data)
                self.network.train()
                utils.reset_learning_rate(self.optimizer,self.lr_setter(step))
                self.optimizer.zero_grad()
                self.network.zero_grad()

                log_info = {}
                output = self.network(train_data)
                loss = self.loss_funcs(output)
                            
                wholeloss+=loss
                loss.backward()
                self.optimizer.step()
                
                if (step+1) % self.train_log_step == 0:
                    loss_info={'loss':wholeloss/self.train_log_step}
                    self._log_data(loss_info,step+1,'train')
                    wholeloss=0

                if (step+1)%self.val_interval==0:
                    val_results=self.val_evaluator(self.network, self.val_set, self.loss_funcs)
                    val_R=val_results[self.val_index]
                    print(f'validation value now: {val_R:.5f}')
                    if self.greater_sign * val_R >= self.greater_sign * self.best_para:
                        print(f'best validation value now: {val_R:.5f} previous {self.best_para:.5f}')
                        self.best_para=val_R
                        self._save_model(step+1,self.best_pth_fn)
                    self._log_data(val_results,step+1,'val')

                if (step+1)%self.save_interval==0:
                    self._save_model(step+1,self.pth_fn)

                pbar.set_postfix(loss=float(loss.detach().cpu().numpy()),lr=self.optimizer.state_dict()['param_groups'][0]['lr'])
                pbar.update(1)
                if step>=whole_step:
                    break
            if step>=whole_step:
                    break
        pbar.close()