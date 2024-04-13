"""
Model in Pytorch of YOHO.
"""

import torch
import torch.nn as nn
import numpy as np


#DRnet
class Comb_Conv(nn.Module):
    def __init__(self,in_dim,out_dim):
        super().__init__()
        self.comb_layer=nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.ReLU(),
            nn.Conv2d(in_dim,out_dim,(1,13),1)
        )
    def forward(self,input):
        return self.comb_layer(input)

class Residual_Comb_Conv(nn.Module):
    def __init__(self,in_dim,middle_dim,out_dim,Nei_in_SO3):
        super().__init__()
        self.Nei_in_SO3=Nei_in_SO3
        self.comb_layer_in=nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.ReLU(),
            nn.Conv2d(in_dim,middle_dim,(1,13),1)
        )
        self.comb_layer_out=nn.Sequential(
            nn.BatchNorm2d(middle_dim),
            nn.ReLU(),
            nn.Conv2d(middle_dim,out_dim,(1,13),1)
        )
        self.short_cut=False
        if not in_dim==out_dim:
            self.short_cut=True
            self.short_cut_layer=nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.ReLU(),
            nn.Conv2d(in_dim,out_dim,(1,13),1)
            )
    
    def data_process(self,data):
        data=torch.squeeze(data)
        if len(data.size())==2:
            data=data[None,:,:]
        data=data[:,:,self.Nei_in_SO3]
        data=torch.reshape(data,[data.shape[0],data.shape[1],60,13])
        return data

    def forward(self,feat_input):#feat:bn*f*60
        feat=self.data_process(feat_input)
        feat=self.comb_layer_in(feat)
        feat=self.data_process(feat)
        feat=self.comb_layer_out(feat)[:,:,:,0]
        if self.short_cut:
            feat_sc=self.data_process(feat_input)
            feat_sc=self.short_cut_layer(feat_sc)[:,:,:,0]
        else:
            feat_sc=feat_input
        
        return feat+feat_sc #output:bn*f*60

class PartI_network(nn.Module):
    def __init__(self, group_dir):
        super().__init__()
        self.group_dir = group_dir
        self.Nei_in_SO3=torch.from_numpy(np.load(f'{self.group_dir}/Nei_Index_in_SO3_ordered_13.npy').astype(np.int).reshape([-1])).cuda()    #nei 60*12 readin
        self.Rgroup_npy=np.load(f'{self.group_dir}/Rotation.npy').astype(np.float32)
        self.Rgroup=torch.from_numpy(self.Rgroup_npy).cuda()

        self.Conv_in=nn.Sequential(nn.Conv2d(32,256,(1,13),1))
        self.SO3_Conv_layers=nn.ModuleList([Residual_Comb_Conv(256,512,256,self.Nei_in_SO3)])
        self.Conv_out=Comb_Conv(256,32)

    def data_process(self,data):
        data=torch.squeeze(data)
        data=data[:,:,self.Nei_in_SO3]
        data=torch.reshape(data,[data.shape[0],data.shape[1],60,13])
        return data

    def SO3_Conv(self,data):#data:bn,f,gn
        data=self.data_process(data)
        data=self.Conv_in(data)[:,:,:,0]
        for layer in range(len(self.SO3_Conv_layers)):
            data=self.SO3_Conv_layers[layer](data)
        data=self.data_process(data)
        data=self.Conv_out(data)[:,:,:,0]
        return data

        
    def forward(self, feats):
        feats_eqv=self.SO3_Conv(feats)# bn,f,gn
        feats_eqv=feats_eqv+feats
        feats_inv=torch.mean(feats_eqv,dim=-1)# bn,f

        #before conv for partII
        feats_eqv=feats_eqv/torch.clamp_min(torch.norm(feats_eqv,dim=1,keepdim=True),min=1e-4)
        feats_inv=feats_inv/torch.clamp_min(torch.norm(feats_inv,dim=1,keepdim=True),min=1e-4)

        return {'inv':feats_inv,'eqv':feats_eqv}

class PartI_test(nn.Module):
    def __init__(self, group_dir):
        super().__init__()
        self.group_dir = group_dir
        self.PartI_net=PartI_network(group_dir)

    def forward(self,group_feat):
        return self.PartI_net(group_feat)

name2network={  
    'PartI_test':PartI_test,
}

