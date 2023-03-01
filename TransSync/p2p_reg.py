import os
import torch
import open3d
import numpy as np
from utils.knn_search import knn_module
from utils.utils import make_non_exists_dir,transform_points

class refiner():
    def __init__(self):
        pass

    def center_cal(self,key_m0,key_m1,scores):
        key_m0=key_m0*scores[:,None]
        key_m1=key_m1*scores[:,None]
        key_m0=np.sum(key_m0,axis=0)
        key_m1=np.sum(key_m1,axis=0)
        return key_m0,key_m1

    def SVDR_w(self,beforerot,afterrot,scores):# beforerot afterrot Scene2,Scene1
        weight=np.diag(scores)
        H=np.matmul(np.matmul(np.transpose(afterrot),weight),beforerot)
        U,Sigma,VT=np.linalg.svd(H)
        return np.matmul(U,VT)

    def R_cal(self,key_m0,key_m1,center0,center1,scores):
        key_m0=key_m0-center0[None,:]
        key_m1=key_m1-center1[None,:]
        return self.SVDR_w(key_m1,key_m0,scores)

    def t_cal(self,center0,center1,R):
        return center0-center1@R.T

    def Rt_cal(self,key_m0,key_m1,scores):
        scores=scores/np.sum(scores)
        center0,center1=self.center_cal(key_m0,key_m1,scores)
        R=self.R_cal(key_m0,key_m1,center0,center1,scores)
        t=self.t_cal(center0,center1,R)
        return R,t
    
    def Refine_trans(self,key_m0,key_m1,T,scores,inlinerdist=None):
        key_m1_t=transform_points(key_m1,T)
        diff=np.sum(np.square(key_m0-key_m1_t),axis=-1)
        overlap=np.where(diff<inlinerdist*inlinerdist)[0]
            
        scores=scores[overlap]
        key_m0=key_m0[overlap]
        key_m1=key_m1[overlap]
        R,t=self.Rt_cal(key_m0, key_m1, scores)
        Tnew=np.eye(4)
        Tnew[0:3,0:3]=R
        Tnew[0:3,3]=t
        return Tnew

class p2preg():
    def __init__(self, cfg):
        self.inlierd = cfg.inlierd
        self.iters = 50000
        self.KNN=knn_module.KNN(1)
        self.refiner = refiner()
            
    def match(self, des0, des1):
        # des0 --> n0*32
        # des1 --> n1*32
        feats0=torch.from_numpy(np.transpose(des0.astype(np.float32))[None,:,:]).cuda()
        feats1=torch.from_numpy(np.transpose(des1.astype(np.float32))[None,:,:]).cuda()
        d,argmin_of_0_in_1=self.KNN(feats1,feats0)
        argmin_of_0_in_1=argmin_of_0_in_1[0,0].cpu().numpy()
        d,argmin_of_1_in_0=self.KNN(feats0,feats1)
        argmin_of_1_in_0=argmin_of_1_in_0[0,0].cpu().numpy()
        match_pps=[]
        for i in range(argmin_of_0_in_1.shape[0]):
            in0=i
            in1=argmin_of_0_in_1[i]
            inv_in0=argmin_of_1_in_0[in1]
            if in0==inv_in0:
                match_pps.append(np.array([[in0,in1]]))
        match_pps=np.concatenate(match_pps,axis=0)
        return match_pps
    
    def inlier_ratio(self, keys0, keys1, match, T):
        Keys_m0 = keys0[match[:,0]]
        Keys_m1 = keys1[match[:,1]]
        Keys_m1 = transform_points(Keys_m1, T)
        inlier = np.sum(np.square(Keys_m0-Keys_m1),axis=-1)
        inlier = np.mean(inlier<self.inlierd*self.inlierd)
        return inlier
    
    def ransac(self, keys0, keys1, match):
        source_pcd = open3d.geometry.PointCloud()
        source_pcd.points = open3d.utility.Vector3dVector(keys0)
        target_pcd = open3d.geometry.PointCloud()
        target_pcd.points = open3d.utility.Vector3dVector(keys1)
        coors = open3d.utility.Vector2iVector(match)
        result = open3d.registration.registration_ransac_based_on_correspondence(
        source_pcd, target_pcd, coors,
        self.inlierd,
        open3d.registration.TransformationEstimationPointToPoint(False), 3,
        open3d.registration.RANSACConvergenceCriteria(self.iters, 1000))            
        trans = result.transformation
        trans = np.linalg.inv(trans)
        
        # refine:
        Keys_m0 = keys0[match[:,0]]
        Keys_m1 = keys1[match[:,1]]
        trans=self.refiner.Refine_trans(Keys_m0,Keys_m1,trans,np.ones(match.shape[0]),inlinerdist=self.inlierd*2.0)
        trans=self.refiner.Refine_trans(Keys_m0,Keys_m1,trans,np.ones(match.shape[0]),inlinerdist=self.inlierd)
        
        # judge overlap after registration
        ir = self.inlier_ratio(keys0, keys1, match, trans)
        
        return trans, ir, match.shape[0]
    
    def get_des(self, dataset, id):
        return 0
    
    def save(self, dataset, id0, id1, T, ir, n_match):
        return 0
    
    def already_exists(self, dataset, id0, id1):
        return False, None, None
    
    def run(self, dataset, id0, id1):
        sign, T, ir, n_matches = self.already_exists(dataset, id0, id1)
        if sign:
            return T, ir, n_matches
        keys0 = dataset.get_kps(id0)
        keys1 = dataset.get_kps(id1)
        des0 = self.get_des(dataset, id0)
        des1 = self.get_des(dataset, id1)
        match = self.match(des0, des1)        
        T, ir, n_matches = self.ransac(keys0,keys1,match)
        self.save(dataset, id0, id1, T, ir, n_matches)
        return T, ir, n_matches


class yoho(p2preg):
    def get_des(self, dataset, sid):
        desdir = f'data/{dataset.name}/yoho_desc'
        des = np.load(f'{desdir}/{sid}.npy')
        return des
    
    def already_exists(self, dataset, id0, id1):
        fn = f'./pre/pairwise_registration/yoho/{dataset.name}/{id0}-{id1}.npz'
        if os.path.exists(fn):
            T = np.load(fn)
            T, ir, n_matches = T['trans'], T['ir'], T['n_matches']
            return True, T, ir, n_matches
        else:
            return False, None, None, None
    
    def save(self, dataset, id0, id1, T, ir, n_matches):
        savedir = f'./pre/pairwise_registration/yoho/{dataset.name}/'
        make_non_exists_dir(savedir)
        np.savez(f'{savedir}/{id0}-{id1}.npz', trans = T, ir = ir, n_matches = n_matches)
        
    def run(self, dataset, id0, id1):
        sign, T, ir, n_matches = self.already_exists(dataset, id0, id1)
        if sign:
            return T, ir, n_matches
        keys0 = dataset.get_kps(id0)
        keys1 = dataset.get_kps(id1)
        des0 = self.get_des(dataset, id0)
        des1 = self.get_des(dataset, id1)
        match = self.match(des0, des1)        
        T, ir, n_matches = self.ransac(keys0,keys1,match)
        self.save(dataset, id0, id1, T, ir, n_matches)
        return T, ir, n_matches

name2estimator={
    'yoho':yoho,
}

        