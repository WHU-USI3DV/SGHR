
import os
import torch
import numpy as np
import open3d as o3d
from glob import glob
from tqdm import tqdm
from Test import tester
from utils.parses import get_config
from TransSync.p2p_reg import p2preg
from utils.utils import transform_points
from yoho.yoho_extract import yoho_extractor
from TransSync.Laplacian_TS import pair2globalT_cycle

class SGHR():
    def __init__(self,
                 fcgf_ckpt='/mnt/proj/Methods/YOHO-master/model/Backbone/best_val_checkpoint.pth',
                 yoho_ckpt='/mnt/proj/Methods/YOHO-master/model/PartI_train/model_best.pth') -> None:
        self.cfg,_ = get_config()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.extractor = yoho_extractor(fcgf_ckpt,yoho_ckpt,device=self.device)
        self.overlap_estimator = tester(self.cfg)

    def _mkdir(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    def _load_pc(self, pc_fn):
        if pc_fn[-3:] == 'npy':
            return np.load(pc_fn)
        else:
            pcd = o3d.io.read_point_cloud(pc_fn)
            pcd = np.array(pcd.points)[:,0:3]
            return pcd

    def _load_pcs(self, pc_dir):
        max_pts_per_pc = 50000
        pc_fns = glob(f'{pc_dir}/*.p*')
        if len(pc_fns)<1: pc_fns = glob(f'{pc_dir}/*.npy')
        pc_fns.sort()
        pcds, frames = [],[]
        for pc_fn in pc_fns:
            frame = str.split(pc_fn,'/')[-1]
            frame = str.split(frame,'.')[-2]
            frames.append(frame)
            pcd = self._load_pc(pc_fn)
            pcd = np.random.permutation(pcd)[0:max_pts_per_pc]
            pcds.append(pcd)
        return pcds, frames

    def _determine_vs(self, pcs):
        delta = 0
        for pc in pcs:
            pc_delta = np.max(pc,axis=0) - np.min(pc,axis=0)
            delta += np.mean(pc_delta)
        delta /= len(pcs)
        vs = delta / 3 * 0.025
        return vs
    
    @torch.no_grad()
    def _extract_yoho(self, pc, nkpts=5000):
        '''
        pc numpy array n*3
        '''
        kpts, yoho_desc, yoho_eqv = self.extractor.run(pc, self.vs, nkpts)
        return kpts, yoho_desc.numpy()

    @torch.no_grad()
    def _determine_overlap(self, kpts, yohos):
        '''
        a list of kpts numpy [5000*3]
        a list of yoho features numpy [5000*32]
        '''
        kpts  = [torch.from_numpy(kpt.astype(np.float32))  for kpt  in kpts ]
        yohos = [torch.from_numpy(yoho.astype(np.float32)) for yoho in yohos]
        batch = {'points':kpts,'feats':yohos}
        overlap = self.overlap_estimator._generate_simmat(batch)
        return overlap
    
    def _multiview_registration(self, kpts, yohos, overlap, topk, n_cycles):
        # registor
        self.pairwise_regor = p2preg(inlierd=self.vs*5)
        
        # determine a sparse graph given the estimated overlaps
        def construct_LSW(overlap, topk):
            # use predicted overlap ratio
            scoremat = overlap
            n,_ = scoremat.shape     
            # keep symmetry
            for i in range(n):
                scoremat[i,i] = 0
                for j in range(i+1,n):
                    scoremat[j,i] = scoremat[i,j]
            # conduct top-k mask
            mask = np.zeros([n,n])
            for i in range(n):
                score_scan = scoremat[i]
                argsort = np.argsort(-score_scan)[:topk]
                mask[i,argsort] = 1
            return scoremat, mask.astype(np.float32)

        # conduct 
        def multiview_registration(scoremat, mask, kpts, yohos, N_cyclegraph):
            # pairwise registration
            n = len(mask)
            Ts = np.zeros([n,n,4,4])
            irs = np.zeros([n,n])
            weights = np.zeros([n,n])
            N_pair = 0
            for i in range(n):
                for j in range(n):
                    if mask[i,j]>0:
                        if i == j:continue
                        # in the following, we must construct a symmetric matrix (weights(if add the noise matrix should also be), Ts) 
                        # for the spectral relaxation solution of rotation synchronization
                        weights[i,j] = 1     
                        weights[j,i] = 1    
                        # If we haven't load the trans and the inv trans, we load the pairwise transformation
                        if np.sum(np.abs(Ts[i,j,0:3,0:3]))<0.001:  
                            # pairwise registration
                            matches = self.pairwise_regor.match(yohos[i],yohos[j])
                            Tij, ir, n_matches = self.pairwise_regor.ransac(kpts[i],kpts[j],matches)
                            # guarantee meaningful rotation matrix
                            if np.linalg.det(Tij[0:3,0:3])<0:
                                Tij[0:2] = Tij[[1,0]]
                            # we use ransac's inlier number/100
                            irs[i,j], irs[j,i] = ir*n_matches/100, ir*n_matches/100
                            Ts[i,j] = Tij
                            Ts[j,i] = np.linalg.inv(Tij)
                            N_pair += 1
            print(f'Estimate {N_pair} pairs')
            # conduct the global transformation syn
            Tglobals,weights_out = pair2globalT_cycle(weights*scoremat*irs, Ts, N_cyclegraph)     
            return Tglobals   

        # pipeline
        scoremat, mask = construct_LSW(overlap,topk)
        pc_poses = multiview_registration(scoremat,mask,kpts,yohos,n_cycles)
        return pc_poses

    def _visual_pcds(self, xyzs, colorize = True, normal = True):
        pcds = []
        for xyz in xyzs:
            if hasattr(xyz,'ndim'):
                xyz = xyz.reshape(-1,3)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(xyz)
                if colorize:
                    pcd.paint_uniform_color(np.random.rand(3))
            else: pcd = xyz
            if normal:
                # determine_nei
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(self.vs*10, 20))
            pcds.append(pcd)
        o3d.visualization.draw_geometries(pcds)

    def run(self, pc_dir, nkpts = 5000, vs = None, topk = None, n_cycles = 50):
        '''
        Input:
        :pc_dir contains a set of .ply/.pcd files for registration
        :nkpts is the number of key points to extract YOHO features
        :vs is the voxel size for pc voxelization, referring https://github.com/HpWang-whu/YOHO/tree/master/others, if set to None, we will auto-set it.   
        :topk means using connect how many pc to a query one in graph construction, if None, we use half of pcs.    
        :n_cycles means IRLS iterations.
        Output:
        :the registration results will be saved to "pc_dir"/registration;
        :"pc_dir"/registration/kpts: sampled kpts of the point clouds in "pc_dir" for yoho feature extraction;
        :"pc_dir"/registration/yoho: yoho features of the point clouds in "pc_dir";
        :"pc_dir"/registration/multi_reg: overlap.txt saving estimated overlap ratio, pose.txt saving calculated pc pose;
        '''
        save_dir = f'{pc_dir}/registration/'
        desc_save_dir = f'{save_dir}/yoho'
        kpts_save_dir = f'{save_dir}/kpts'
        pose_save_dir = f'{save_dir}/multi_reg'
        self._mkdir(desc_save_dir)
        self._mkdir(kpts_save_dir)
        self._mkdir(pose_save_dir)
        # load pcs
        pcds, frames = self._load_pcs(pc_dir)
        if vs is None:
            self.vs = self._determine_vs(pcds)
        else:
            self.vs = vs
        # extract yoho features
        print('Extracting YOHO features...')
        kpts, yohos = [],[]
        for i in tqdm(range(len(pcds))):
            if os.path.exists(f'{desc_save_dir}/{frames[i]}.desc.npy'):
                kpt  = np.load(f'{kpts_save_dir}/{frames[i]}.kpts.npy')
                yoho = np.load(f'{desc_save_dir}/{frames[i]}.desc.npy')
            else:
                kpt, yoho = self._extract_yoho(pcds[i],nkpts)
                np.save(f'{kpts_save_dir}/{frames[i]}.kpts.npy',kpt)
                np.save(f'{desc_save_dir}/{frames[i]}.desc.npy',yoho)
            kpts.append(kpt)
            yohos.append(yoho)
        # estimated overlap
        print('Conducting SGHR overlap estimation...')
        overlap = self._determine_overlap(kpts, yohos)
        np.savetxt(f'{pose_save_dir}/overlap.txt',overlap)
        # multiview registration
        print('Conducting SGHR pose estimation...')
        if topk is None:
            topk = max(3,int(len(pcds)/2.))
        poses = self._multiview_registration(kpts,yohos,overlap,topk,n_cycles)
        for i, pose in enumerate(poses):
            np.savetxt(f'{pose_save_dir}/{frames[i]}.pose.txt',pose)
        # visual
        posed_pcds = [transform_points(pcds[i],poses[i]) for i in range(len(pcds))]
        self._visual_pcds(posed_pcds)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pcdir',default="data/demo",type=str,help='dir of pc files in .pcd/.ply/.npy')
    # The following parameters do not need setting if you just want to use SGHR on-the-fly.
    parser.add_argument('--bkbckpt',default='yoho/ckpts/backbone/best_val_checkpoint.pth',type=str)
    parser.add_argument('--yohockpt',default='yoho/ckpts/yoho/model_best.pth',type=str)
    parser.add_argument('--nkpts',default=3000,type=int,help='extract 3k kpts on each pc')
    parser.add_argument('--vs',default=None,type=float,help='voxel size for pc voxelization, referring https://github.com/HpWang-whu/YOHO/tree/master/others,\
                                                             if set to None, we will auto-set it.')
    parser.add_argument('--topk',default=None,type=int,help='connect how many pc to a query pc in graph construction, if None, we use half of len(pcs).')
    parser.add_argument('--ncycles',default=100,type=int,help='n_cycles means IRLS iterations')
    args = parser.parse_args()

    # for registration
    regor = SGHR(fcgf_ckpt=args.bkbckpt,yoho_ckpt=args.yohockpt)
    regor.run(pc_dir=args.pcdir,
              nkpts=args.nkpts,
              vs=args.vs,
              topk=args.topk,
              n_cycles=args.ncycles)



