##########################
# input: target-1*feature_dimension*n, source-1*feature_dimension*m
# output: nearest_dist_of_source_in_target-1*k*m, nearest_index_of_source_in_target-1*k*m

# knn_search.knn_module.KNN -- output a matcher with k as the global parameter
# the matcher is callable for KNN search

##########################

import torch
import numpy as np

class modified_knn_matcher():
    def __init__(self, k = 1) -> None:
        self.k = k

    def pdist(self, A, B, dist_type='L2'):
          if dist_type == 'L2':
              D2 = torch.sum((A.unsqueeze(1) - B.unsqueeze(0)).pow(2), 2)
              return torch.sqrt(D2 + 1e-7)
          elif dist_type == 'SquareL2':
              return torch.sum((A.unsqueeze(1) - B.unsqueeze(0)).pow(2), 2)
          else:
              raise NotImplementedError('Not implemented')

    def find_nn_gpu(self, 
                    source_F, 
                    target_F, 
                    nn_max_n=1000, 
                    return_distance=True, 
                    dist_type='SquareL2'):
        # Too much memory if F0 or F1 large. Divide the F0
        F0 = source_F.squeeze()
        F1 = target_F.squeeze()
        if nn_max_n > 1:
            N = len(F0)
            C = int(np.ceil(N / nn_max_n))
            stride = nn_max_n
            dists, inds = [], []
            for i in range(C):
                dist = self.pdist(F0[i * stride:(i + 1) * stride], F1, dist_type=dist_type)
                min_dist, ind = dist.min(dim=1)
                dists.append(min_dist.detach().unsqueeze(1).cpu())
                inds.append(ind.cpu())

            if C * stride < N:
                dist = self.pdist(F0[C * stride:], F1, dist_type=dist_type)
                min_dist, ind = dist.min(dim=1)
                dists.append(min_dist.detach().unsqueeze(1).cpu())
                inds.append(ind.cpu())

            dists = torch.cat(dists)
            inds = torch.cat(inds)
            assert len(inds) == N
        else:
            dist = self.pdist(F0, F1, dist_type=dist_type)
            min_dist, inds = dist.min(dim=1)
            dists = min_dist.detach().unsqueeze(1).cpu()
            inds = inds.cpu()
        # output: source_F with the dimension of m*f, inds with m, dists with m
        dists = dists.squeeze()
        inds = inds.squeeze()
        if return_distance:
            return dists, inds
        else:
            return inds

    def find_knn_gpu(self, 
                    source_F, 
                    target_F, 
                    nn_max_n=1000, 
                    return_distance=True, 
                    dist_type='SquareL2'):
        F0 = source_F.squeeze()
        F1 = target_F.squeeze()
        # Too much memory if F0 or F1 large. Divide the F0
        if nn_max_n > 1:
            N = len(F0)
            C = int(np.ceil(N / nn_max_n))
            stride = nn_max_n
            dists, inds = [], []
            for i in range(C):
                dist = self.pdist(F0[i * stride:(i + 1) * stride], F1, dist_type=dist_type)
                min_dist, ind = torch.topk(-dist, self.k, dim=1)
                dists.append(-min_dist.detach().unsqueeze(1).cpu())
                inds.append(ind.cpu())

            if C * stride < N:
                dist = self.pdist(F0[C * stride:], F1, dist_type=dist_type)
                min_dist, ind = torch.topk(-dist, self.k, dim=1)
                dists.append(-min_dist.detach().unsqueeze(1).cpu())
                inds.append(ind.cpu())

            dists = torch.cat(dists,dim=0)
            inds = torch.cat(inds,dim=0)
            assert len(inds) == N
        else:
            dist = self.pdist(F0, F1, dist_type=dist_type)
            min_dist, inds = torch.topk(-dist, self.k, dim=1)
            dists = -min_dist.detach().unsqueeze(1).cpu()
            inds = inds.cpu()
        # output: source_F with the dimension of m*f, inds with m*self.k, dists with m*self.k
        if return_distance:
            return dists, inds
        else:
            return inds
    
    def find_corr(self, 
                  F0, 
                  F1, 
                  subsample_size=-1, 
                  mutual = True, 
                  nn_max_n = 500):
        #init
        inds0, inds1 = np.arange(F0.shape[0]), np.arange(F1.shape[0])
        if subsample_size > 0:
            N0 = min(len(F0), subsample_size)
            N1 = min(len(F1), subsample_size)
            inds0 = np.random.choice(len(F0), N0, replace=False)
            inds1 = np.random.choice(len(F1), N1, replace=False)
            F0 = F0[inds0]
            F1 = F1[inds1]
        # Compute the nn
        nn_inds_in1 = self.find_nn_gpu(F0, F1, nn_max_n=nn_max_n, return_distance=False)
        if not mutual:
          inds1 = inds1[nn_inds_in1]
        else:
          matches = []
          nn_inds_in0 = self.find_nn_gpu(F1, F0, nn_max_n=nn_max_n, return_distance=False)
          for i in range(len(nn_inds_in1)):
              if i == nn_inds_in0[nn_inds_in1[i]]:
                matches.append((i, nn_inds_in1[i]))
          matches = np.array(matches).astype(np.int32)
          inds0 = inds0[matches[:,0]]
          inds1 = inds1[matches[:,1]]
        return inds0, inds1

    def __call__(self, 
                 target_F, 
                 source_F, 
                 nn_max_n=500, 
                 dist_type='L2'):
        # to keep consistent with the original knn module
        # target_F: 1*f*n and source_F: 1*f*m
        target_F = target_F.squeeze().T
        source_F = source_F.squeeze().T
        if self.k<2:
            # d,index with dimension of m -> 1*1*m
            d,idx = self.find_nn_gpu(source_F=source_F,
                                    target_F=target_F,
                                    nn_max_n=nn_max_n,
                                    return_distance=True,
                                    dist_type=dist_type)
            return d[None,None],idx[None,None]
        else:
            # d,index with dimension of m*k -> 1*k*m
            d,idx = self.find_knn_gpu(source_F=source_F,
                                     target_F=target_F,
                                     nn_max_n=nn_max_n,
                                     return_distance=True,
                                     dist_type=dist_type)
            return d.T[None],idx.T[None]
            

class knn_module_class():
    def __init__(self) -> None:
        pass

    def KNN(self,k):
        return modified_knn_matcher(k)

knn_module = knn_module_class()