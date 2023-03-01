import utils.knn_search.knn_search as _knn
import torch
import torch.nn as nn


def knn(ref, query, k):
    d, i = _knn.knn_search(ref, query, k)
    i -= 1
    return d, i

class KNN(nn.Module):
    def __init__(self, k, transpose_mode=False):
        super(KNN, self).__init__()
        self.k = k
        self._t = transpose_mode

    @staticmethod
    def _T(t, mode=False):
        if mode:
            return t.transpose(0, 1).contiguous()
        else:
            return t.contiguous()

    def forward(self, ref, query):
        assert ref.size(0) == query.size(0), "ref.shape={} != query.shape={}".format(ref.shape, query.shape)
        with torch.no_grad():
            batch_size = ref.size(0)
            D, I = [], []
            for bi in range(batch_size):
                r, q = self._T(ref[bi], self._t), self._T(query[bi], self._t)
                d, i = knn(r.float(), q.float(), self.k)
                d, i = self._T(d, self._t), self._T(i, self._t)
                D.append(d)
                I.append(i)
            D = torch.stack(D, dim=0)
            I = torch.stack(I, dim=0)
        return D, I
